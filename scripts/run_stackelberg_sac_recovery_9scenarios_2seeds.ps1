param(
    [int[]]$Seeds = @(0, 1),
    [string]$OutputName = "stackelberg_sac_from_sac_nominal_9scenarios_2seeds_20260628",
    [int]$MaxParallel = 2,
    [int]$TorchThreadsPerProcess = 2,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [switch]$ForcePrepare,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PrepareScript = Join-Path $ScriptDir "prepare_stackelberg_sac_recovery_from_baseline.py"
$TailScript = Join-Path $ScriptDir "continue_shock_recovery_tail.py"
$OutputRoot = Join-Path $Root (Join-Path "runs\sac_modified" $OutputName)
$LogRoot = Join-Path $OutputRoot "logs"

$Scenarios = @(
    "level1_easy",
    "level2_easy",
    "level3_easy",
    "level1_medium",
    "level2_medium",
    "level3_medium",
    "level1_hard",
    "level2_hard",
    "level3_hard"
)

function Test-CompleteSeedDir {
    param([string]$SeedDir)
    $finalCkpt = Join-Path $SeedDir ("checkpoints\checkpoint_recovery_step_{0:D5}.pt" -f $RecoveryTimesteps)
    return (
        (Test-Path -LiteralPath (Join-Path $SeedDir "shock_recovery_summary.csv")) -and
        (Test-Path -LiteralPath (Join-Path $SeedDir "shock_recovery_curve.csv")) -and
        (Test-Path -LiteralPath (Join-Path $SeedDir "checkpoints\checkpoint_nominal.pt")) -and
        (Test-Path -LiteralPath $finalCkpt)
    )
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing Python executable: $Python"
}
if (-not (Test-Path -LiteralPath $PrepareScript)) {
    throw "Missing prepare script: $PrepareScript"
}
if (-not (Test-Path -LiteralPath $TailScript)) {
    throw "Missing tail script: $TailScript"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:MKL_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:OPENBLAS_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:NUMEXPR_NUM_THREADS = [string]$TorchThreadsPerProcess

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$SeedArgs = @()
foreach ($seed in $Seeds) {
    $SeedArgs += [string]$seed
}
$prepareArgs = @(
    $PrepareScript,
    "--output-root", $OutputRoot,
    "--seeds"
) + $SeedArgs + @(
    "--python", $Python,
    "--device", "cpu",
    "--recovery-timesteps", [string]$RecoveryTimesteps,
    "--eval-interval", [string]$EvalInterval,
    "--num-eval-episodes", [string]$NumEvalEpisodes
)
if ($ForcePrepare) {
    $prepareArgs += "--force"
}

Write-Host ("Preparing Stackelberg-SAC recovery dirs under {0}" -f $OutputRoot)
& $Python @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "Prepare script failed with exit code $LASTEXITCODE"
}

$Jobs = New-Object System.Collections.Generic.List[object]
foreach ($scenario in $Scenarios) {
    foreach ($seed in $Seeds) {
        $SeedDir = Join-Path $OutputRoot (Join-Path $scenario ("seed" + $seed))
        if (Test-CompleteSeedDir -SeedDir $SeedDir) {
            Write-Host ("Skipping complete Stackelberg-SAC {0} seed{1}: {2}" -f $scenario, $seed, $SeedDir)
            continue
        }
        $Jobs.Add([pscustomobject]@{
            Scenario = $scenario
            Seed = $seed
            SeedDir = $SeedDir
            Stdout = Join-Path $LogRoot ("{0}_seed{1}_stdout.log" -f $scenario, $seed)
            Stderr = Join-Path $LogRoot ("{0}_seed{1}_stderr.log" -f $scenario, $seed)
        })
    }
}

Write-Host ("Queued {0} incomplete Stackelberg-SAC recovery jobs with MaxParallel={1}" -f $Jobs.Count, $MaxParallel)
if ($DryRun) {
    foreach ($job in $Jobs) {
        Write-Host ("DRYRUN {0} --seed-dir {1}" -f $TailScript, $job.SeedDir)
    }
    exit 0
}

$Pending = [System.Collections.Queue]::new()
foreach ($job in $Jobs) {
    $Pending.Enqueue($job)
}
$Running = New-Object System.Collections.Generic.List[object]
$SummaryRows = New-Object System.Collections.Generic.List[object]
$Failed = $false

function Start-NextJobs {
    while (($Running.Count -lt $MaxParallel) -and ($Pending.Count -gt 0)) {
        $job = $Pending.Dequeue()
        $argsList = @(
            $TailScript,
            "--seed-dir", $job.SeedDir,
            "--target-step", [string]$RecoveryTimesteps,
            "--python", $Python,
            "--device", "cpu"
        )
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList $argsList `
            -WorkingDirectory $Root `
            -PassThru `
            -WindowStyle Hidden `
            -RedirectStandardOutput $job.Stdout `
            -RedirectStandardError $job.Stderr
        $Running.Add([pscustomobject]@{
            Process = $proc
            Scenario = $job.Scenario
            Seed = $job.Seed
            SeedDir = $job.SeedDir
            Stdout = $job.Stdout
            Stderr = $job.Stderr
            StartTime = Get-Date
        })
        Write-Host ("Started Stackelberg-SAC {0} seed{1}: PID={2}" -f $job.Scenario, $job.Seed, $proc.Id)
    }
}

Start-NextJobs
while ($Running.Count -gt 0) {
    Start-Sleep -Seconds 30
    $StillRunning = New-Object System.Collections.Generic.List[object]
    foreach ($item in $Running) {
        if ($item.Process.HasExited) {
            $exitCode = [int]$item.Process.ExitCode
            $durationMin = ((Get-Date) - $item.StartTime).TotalMinutes
            $complete = Test-CompleteSeedDir -SeedDir $item.SeedDir
            Write-Host ("Finished Stackelberg-SAC {0} seed{1}: exit={2} complete={3} duration={4:N1}min" -f $item.Scenario, $item.Seed, $exitCode, $complete, $durationMin)
            $SummaryRows.Add([pscustomobject]@{
                scenario = $item.Scenario
                seed = $item.Seed
                exit_code = $exitCode
                complete = $complete
                duration_min = [math]::Round($durationMin, 2)
                seed_dir = $item.SeedDir
                stdout = $item.Stdout
                stderr = $item.Stderr
            })
            if (($exitCode -ne 0) -or (-not $complete)) {
                $Failed = $true
                Write-Host ("Check stderr: {0}" -f $item.Stderr)
            }
        } else {
            $StillRunning.Add($item)
        }
    }
    $Running = $StillRunning
    Start-NextJobs
}

$SummaryPath = Join-Path $OutputRoot ("stackelberg_sac_recovery_run_summary_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$SummaryRows | Export-Csv -NoTypeInformation -Path $SummaryPath
Write-Host ("Summary: {0}" -f $SummaryPath)

if ($Failed) {
    Write-Host "Completed with at least one failed or incomplete job."
    exit 1
}

Write-Host ("All Stackelberg-SAC recovery jobs complete: {0}" -f $OutputRoot)
