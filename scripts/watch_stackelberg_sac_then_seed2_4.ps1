param(
    [string]$OutputName = "stackelberg_sac_from_sac_nominal_9scenarios_2seeds_20260628",
    [int]$PreviousLauncherPid = 0,
    [int[]]$NextSeeds = @(2, 3, 4),
    [int]$MaxParallel = 2,
    [int]$TorchThreadsPerProcess = 2,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [int]$PollSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$LauncherScript = Join-Path $ScriptDir "run_stackelberg_sac_recovery_9scenarios_2seeds.ps1"
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

function Get-Seed01CompleteCount {
    $count = 0
    foreach ($scenario in $Scenarios) {
        foreach ($seed in @(0, 1)) {
            $seedDir = Join-Path $OutputRoot (Join-Path $scenario ("seed" + $seed))
            if (Test-CompleteSeedDir -SeedDir $seedDir) {
                $count += 1
            }
        }
    }
    return $count
}

if (-not (Test-Path -LiteralPath $LauncherScript)) {
    throw "Missing launcher script: $LauncherScript"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Write-Host ("Watcher started at {0}" -f (Get-Date -Format "s"))
Write-Host ("Waiting for seed0/seed1 completion under {0}" -f $OutputRoot)
if ($PreviousLauncherPid -gt 0) {
    Write-Host ("Previous launcher PID: {0}" -f $PreviousLauncherPid)
}

while ($true) {
    $completeCount = Get-Seed01CompleteCount
    $previousRunning = $false
    if ($PreviousLauncherPid -gt 0) {
        $previousRunning = $null -ne (Get-Process -Id $PreviousLauncherPid -ErrorAction SilentlyContinue)
    }
    Write-Host ("{0} complete_seed0_1={1}/18 previous_running={2}" -f (Get-Date -Format "s"), $completeCount, $previousRunning)
    if (($completeCount -eq 18) -and (-not $previousRunning)) {
        break
    }
    Start-Sleep -Seconds ([Math]::Max($PollSeconds, 30))
}

$seedArgs = @()
foreach ($seed in $NextSeeds) {
    $seedArgs += [string]$seed
}
$stdout = Join-Path $LogRoot "launcher_seed2_4_stdout.log"
$stderr = Join-Path $LogRoot "launcher_seed2_4_stderr.log"
$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $LauncherScript,
    "-Seeds"
) + $seedArgs + @(
    "-OutputName",
    $OutputName,
    "-MaxParallel",
    [string]$MaxParallel,
    "-TorchThreadsPerProcess",
    [string]$TorchThreadsPerProcess,
    "-RecoveryTimesteps",
    [string]$RecoveryTimesteps,
    "-EvalInterval",
    [string]$EvalInterval,
    "-NumEvalEpisodes",
    [string]$NumEvalEpisodes
)

$proc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

Write-Host ("Started seed2-4 Stackelberg-SAC launcher at {0}: PID={1}" -f (Get-Date -Format "s"), $proc.Id)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
