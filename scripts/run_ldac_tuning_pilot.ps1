$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PrepareScript = Join-Path $Root "scripts\prepare_ldac_tuning_pilot.py"
$ContinueScript = Join-Path $Root "scripts\continue_shock_recovery_tail.py"
$OutRoot = Join-Path $Root "runs\sac_modified\ldac_tuning_pilot_20260624"
$LogRoot = Join-Path $OutRoot "logs"
$MaxParallel = 4

function Get-CpuAffinityMask {
    $logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $limit = [Math]::Max(1, [Math]::Floor($logical * 0.75))
    $mask = [int64]0
    for ($i = 0; $i -lt $limit; $i++) {
        $mask = $mask -bor ([int64]1 -shl $i)
    }
    return [pscustomobject]@{
        Mask = $mask
        Limit = $limit
        Logical = $logical
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

& $Python $PrepareScript --output-root $OutRoot
if ($LASTEXITCODE -ne 0) {
    throw "prepare_ldac_tuning_pilot.py failed with exit code $LASTEXITCODE"
}

$Affinity = Get-CpuAffinityMask
try {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$Affinity.Mask
    Write-Host ("Scheduler CPU affinity: {0}/{1} logical cores" -f $Affinity.Limit, $Affinity.Logical)
} catch {
    Write-Host "Scheduler CPU affinity could not be set."
}

$JobsPath = Join-Path $OutRoot "tuning_jobs.csv"
$jobs = Import-Csv $JobsPath
$running = @()
$queue = New-Object System.Collections.Queue

foreach ($job in $jobs) {
    $target = [int]$job.target_step
    $doneCheckpoint = Join-Path $job.seed_dir ("checkpoints\checkpoint_recovery_step_{0:D5}.pt" -f $target)
    if (Test-Path $doneCheckpoint) {
        Write-Host ("Skipping completed {0} {1} seed{2}: target={3}" -f $job.candidate, $job.scenario, $job.seed, $target)
        continue
    }
    $activeProcess = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object {
            $_.CommandLine -like "*continue_shock_recovery_tail.py*" -and
            $_.CommandLine -like ("*" + [string]$job.seed_dir + "*")
        } |
        Select-Object -First 1
    if ($activeProcess -ne $null) {
        $proc = Get-Process -Id $activeProcess.ProcessId -ErrorAction SilentlyContinue
        if ($proc -ne $null) {
            $running += [pscustomobject]@{
                Process = $proc
                Candidate = $job.candidate
                Scenario = $job.scenario
                Seed = $job.seed
                SeedDir = $job.seed_dir
            }
            Write-Host ("Resuming active {0} {1} seed{2}: PID={3}" -f $job.candidate, $job.scenario, $job.seed, $proc.Id)
            continue
        }
    }
    $queue.Enqueue($job)
}

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $job = $queue.Dequeue()
        $logPrefix = "{0}_{1}_seed{2}" -f $job.candidate, $job.scenario, $job.seed
        $stdout = Join-Path $LogRoot ($logPrefix + "_stdout.log")
        $stderr = Join-Path $LogRoot ($logPrefix + "_stderr.log")
        $argsList = @(
            $ContinueScript,
            "--seed-dir", $job.seed_dir,
            "--target-step", $job.target_step,
            "--python", $Python,
            "--device", "cpu"
        )
        $proc = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        try {
            $proc.ProcessorAffinity = [IntPtr]$Affinity.Mask
        } catch {
            Write-Host ("Could not set CPU affinity for PID={0}" -f $proc.Id)
        }
        $running += [pscustomobject]@{
            Process = $proc
            Candidate = $job.candidate
            Scenario = $job.scenario
            Seed = $job.seed
            SeedDir = $job.seed_dir
        }
        Write-Host ("Started {0} {1} seed{2}: PID={3}, target={4}" -f $job.candidate, $job.scenario, $job.seed, $proc.Id, $job.target_step)
    }

    Start-Sleep -Seconds 30
    $stillRunning = @()
    foreach ($item in $running) {
        if ($item.Process.HasExited) {
            Write-Host ("Finished {0} {1} seed{2}: exit={3}" -f $item.Candidate, $item.Scenario, $item.Seed, $item.Process.ExitCode)
        } else {
            $stillRunning += $item
        }
    }
    $running = $stillRunning
}

Write-Host ("All LDAC tuning pilot jobs finished: {0}" -f $OutRoot)
