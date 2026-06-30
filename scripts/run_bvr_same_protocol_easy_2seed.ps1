$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runner = Join-Path $Root "run_shock_recovery_experiment.py"
$OutRoot = Join-Path $Root "runs\bvr\same_protocol_easy_2seed_gate_20260620"
$LogRoot = Join-Path $OutRoot "logs"
$MaxParallel = 3

$Experiments = @(
    @{ Level = "level1_easy"; Config = "configs\levels\ppo_difficulty\level1_easy.json"; EvalEpisodes = 300 },
    @{ Level = "level2_easy"; Config = "configs\levels\ppo_difficulty\level2_easy.json"; EvalEpisodes = 128 },
    @{ Level = "level3_easy"; Config = "configs\levels\ppo_difficulty\level3_easy.json"; EvalEpisodes = 128 }
)

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$queue = New-Object System.Collections.Queue
foreach ($exp in $Experiments) {
    foreach ($seed in 0, 1) {
        $queue.Enqueue(@{
            Level = $exp.Level
            Config = $exp.Config
            EvalEpisodes = $exp.EvalEpisodes
            Seed = $seed
        })
    }
}

$running = @()
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $job = $queue.Dequeue()
        $jobOut = Join-Path $OutRoot (Join-Path $job.Level ("seed" + $job.Seed))
        $logPrefix = "{0}_seed{1}" -f $job.Level, $job.Seed
        $stdout = Join-Path $LogRoot ($logPrefix + "_stdout.log")
        $stderr = Join-Path $LogRoot ($logPrefix + "_stderr.log")
        $argsList = @(
            $Runner,
            "--algo", "ppo",
            "--level-config", (Join-Path $Root $job.Config),
            "--output-dir", $jobOut,
            "--nominal-timesteps", "50000",
            "--recovery-timesteps", "20480",
            "--eval-interval", "1024",
            "--num-eval-episodes", [string]$job.EvalEpisodes,
            "--train-eval-episodes", "64",
            "--seed", [string]$job.Seed,
            "--in-domain-seed", "909",
            "--heldout-seed", "1919",
            "--python", $Python,
            "--device", "cpu",
            "--clean-output",
            "--bvr-recovery-enabled"
        )
        $proc = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $running += [pscustomobject]@{
            Process = $proc
            Level = $job.Level
            Seed = $job.Seed
            Output = $jobOut
        }
        Write-Host ("Started {0} seed{1}: PID={2}" -f $job.Level, $job.Seed, $proc.Id)
    }

    Start-Sleep -Seconds 30
    $stillRunning = @()
    foreach ($item in $running) {
        if ($item.Process.HasExited) {
            Write-Host ("Finished {0} seed{1}: exit={2}" -f $item.Level, $item.Seed, $item.Process.ExitCode)
        } else {
            $stillRunning += $item
        }
    }
    $running = $stillRunning
}

Write-Host ("All BVR same-protocol easy 2-seed experiments finished: {0}" -f $OutRoot)
