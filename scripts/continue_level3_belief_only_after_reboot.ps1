param(
    [int]$MaxParallel = 3,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "cpu",
    [int]$TargetStep = 20480
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $Root "run_shock_recovery_experiment.py"
$TailRunner = Join-Path $Root "scripts\continue_shock_recovery_tail.py"
$BaseConfig = Join-Path $Root "configs\ppo_lunar_viper_relative_reward.json"
$LaunchLogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LaunchLogDir | Out-Null
$LaunchLog = Join-Path $LaunchLogDir ("level3_belief_only_reboot_resume_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-LaunchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $LaunchLog -Value $line
}

function Test-SeedComplete {
    param([string]$SeedDir)
    return (
        (Test-Path (Join-Path $SeedDir "shock_recovery_summary.csv")) -and
        (Test-Path (Join-Path $SeedDir "shock_recovery_curve.csv")) -and
        (Test-Path (Join-Path $SeedDir "checkpoints\checkpoint_nominal.pt")) -and
        (Test-Path (Join-Path $SeedDir ("checkpoints\checkpoint_recovery_step_{0:d5}.pt" -f $TargetStep)))
    )
}

function Test-CanTailResume {
    param([string]$SeedDir)
    if (-not (Test-Path (Join-Path $SeedDir "run_config.json"))) {
        return $false
    }
    if (-not (Test-Path (Join-Path $SeedDir "checkpoints\checkpoint_nominal.pt"))) {
        return $false
    }
    $checkpoints = Get-ChildItem -LiteralPath (Join-Path $SeedDir "checkpoints") -Filter "checkpoint_recovery_step_*.pt" -File -ErrorAction SilentlyContinue
    return @($checkpoints).Count -gt 0
}

function New-FreshJob {
    param(
        [string]$Kind,
        [string]$Algo,
        [string]$Level,
        [string]$Config,
        [int]$Seed,
        [string]$Output,
        [switch]$PrSac
    )
    [pscustomobject]@{
        Mode = "fresh"
        Kind = $Kind
        Algo = $Algo
        Level = $Level
        Config = $Config
        Seed = $Seed
        Output = $Output
        PrSac = [bool]$PrSac
    }
}

function New-TailJob {
    param(
        [string]$Kind,
        [string]$Algo,
        [string]$Level,
        [int]$Seed,
        [string]$Output
    )
    [pscustomobject]@{
        Mode = "tail"
        Kind = $Kind
        Algo = $Algo
        Level = $Level
        Config = ""
        Seed = $Seed
        Output = $Output
        PrSac = $false
    }
}

function Add-IfIncomplete {
    param(
        [System.Collections.Generic.List[object]]$Jobs,
        [object]$Job
    )
    if (Test-SeedComplete -SeedDir $Job.Output) {
        Write-LaunchLog ("Skipping complete {0} {1} seed{2}: {3}" -f $Job.Kind, $Job.Level, $Job.Seed, $Job.Output)
        return
    }
    $Jobs.Add($Job)
}

$levelConfigs = @(
    @{ Level = "level3_easy"; Config = "configs\levels\ppo_difficulty\level3_easy.json" },
    @{ Level = "level3_medium"; Config = "configs\levels\ppo_difficulty\level3_medium.json" },
    @{ Level = "level3_hard"; Config = "configs\levels\ppo_difficulty\level3_hard.json" }
)

$jobs = New-Object System.Collections.Generic.List[object]

foreach ($algo in @("ppo", "sac")) {
    foreach ($level in $levelConfigs) {
        foreach ($seed in 0..4) {
            $outRoot = Join-Path $Root ("runs\rl_baselines\{0}\{1}_shock_recovery_5seeds" -f $algo, $level.Level)
            $output = Join-Path $outRoot ("seed{0}" -f $seed)
            if ($algo -eq "sac" -and $level.Level -eq "level3_hard" -and $seed -eq 3 -and (Test-CanTailResume -SeedDir $output)) {
                Add-IfIncomplete -Jobs $jobs -Job (New-TailJob -Kind "rl_baseline" -Algo $algo -Level $level.Level -Seed $seed -Output $output)
            } else {
                Add-IfIncomplete -Jobs $jobs -Job (New-FreshJob -Kind "rl_baseline" -Algo $algo -Level $level.Level -Config $level.Config -Seed $seed -Output $output)
            }
        }
    }
}

$ldacRoots = @{
    "level3_easy:0" = "runs\sac_modified\ldac_easy_seed0_20260621"
    "level3_easy:1" = "runs\sac_modified\ldac_v2_easy_seed1_2_20260621"
    "level3_easy:2" = "runs\sac_modified\ldac_v2_easy_seed1_2_20260621"
    "level3_easy:3" = "runs\sac_modified\ldac_v2_easy_seed3_4_20260623"
    "level3_easy:4" = "runs\sac_modified\ldac_v2_easy_seed3_4_20260623"
    "level3_medium:0" = "runs\sac_modified\ldac_v2_medium_seed0_1_20260621"
    "level3_medium:1" = "runs\sac_modified\ldac_v2_medium_seed0_1_20260621"
    "level3_medium:2" = "runs\sac_modified\ldac_v2_medium_seed2_4_20260623"
    "level3_medium:3" = "runs\sac_modified\ldac_v2_medium_seed2_4_20260623"
    "level3_medium:4" = "runs\sac_modified\ldac_v2_medium_seed2_4_20260623"
    "level3_hard:0" = "runs\sac_modified\ldac_v2_hard_seed0_1_20260621"
    "level3_hard:1" = "runs\sac_modified\ldac_v2_hard_seed0_1_20260621"
    "level3_hard:2" = "runs\sac_modified\ldac_v2_hard_seed2_4_20260623"
    "level3_hard:3" = "runs\sac_modified\ldac_v2_hard_seed2_4_20260623"
    "level3_hard:4" = "runs\sac_modified\ldac_v2_hard_seed2_4_20260623"
}

foreach ($level in $levelConfigs) {
    foreach ($seed in 0..4) {
        $rootKey = "{0}:{1}" -f $level.Level, $seed
        $outRoot = Join-Path $Root $ldacRoots[$rootKey]
        $output = Join-Path (Join-Path $outRoot $level.Level) ("seed{0}" -f $seed)
        Add-IfIncomplete -Jobs $jobs -Job (New-FreshJob -Kind "pr_sac" -Algo "sac" -Level $level.Level -Config $level.Config -Seed $seed -Output $output -PrSac)
    }
}

$queue = New-Object System.Collections.Queue
foreach ($job in $jobs) {
    $queue.Enqueue($job)
}

$running = New-Object System.Collections.Generic.List[object]
$completed = New-Object System.Collections.Generic.List[object]
Write-LaunchLog ("Queued {0} incomplete Level 3 belief-only jobs with MaxParallel={1}" -f $jobs.Count, $MaxParallel)

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $job = $queue.Dequeue()
        $outputRoot = Split-Path -Parent $job.Output
        $logRoot = Join-Path $outputRoot "logs"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $logPrefix = "{0}_{1}_seed{2}_{3}" -f $job.Kind, $job.Level, $job.Seed, $job.Mode
        $stdout = Join-Path $logRoot ($logPrefix + "_stdout.log")
        $stderr = Join-Path $logRoot ($logPrefix + "_stderr.log")

        if ($job.Mode -eq "tail") {
            $argsList = @(
                "-u",
                $TailRunner,
                "--seed-dir", $job.Output,
                "--target-step", [string]$TargetStep,
                "--python", $Python,
                "--device", $Device
            )
        } else {
            $argsList = @(
                $Runner,
                "--algo", $job.Algo,
                "--level-config", (Join-Path $Root $job.Config),
                "--base-config", $BaseConfig,
                "--output-dir", $job.Output,
                "--nominal-timesteps", "50000",
                "--recovery-timesteps", [string]$TargetStep,
                "--eval-interval", "1024",
                "--num-eval-episodes", "128",
                "--train-eval-episodes", "64",
                "--seed", [string]$job.Seed,
                "--in-domain-seed", "909",
                "--heldout-seed", "1919",
                "--python", $Python,
                "--device", $Device,
                "--clean-output"
            )

            if ($job.PrSac) {
                $argsList += @(
                    "--sac-game-recovery-enabled",
                    "--sac-game-anchor-coef", "0.40",
                    "--sac-game-advantage-coef", "0.12",
                    "--sac-game-q-margin", "0.02",
                    "--sac-game-gate-temperature", "0.05",
                    "--sac-game-anchor-barrier-coef", "1.5",
                    "--sac-game-anchor-radius", "0.12",
                    "--sac-recovery-deterministic-actor-update",
                    "--sac-recovery-fixed-alpha", "0.015",
                    "--sac-recovery-target-entropy-scale", "0.05",
                    "--sac-recovery-rollout-deterministic-prob", "0.85",
                    "--sac-recovery-rollout-noise-std", "0.015",
                    "--sac-recovery-log-std-penalty-coef", "0.01",
                    "--sac-recovery-log-std-target", "-2.0"
                )
            }
        }

        $proc = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $running.Add([pscustomobject]@{
            Process = $proc
            Job = $job
            Stdout = $stdout
            Stderr = $stderr
            Started = Get-Date
        })
        Write-LaunchLog ("Started PID={0} mode={1} {2} {3} seed{4}" -f $proc.Id, $job.Mode, $job.Kind, $job.Level, $job.Seed)
    }

    Start-Sleep -Seconds 20

    for ($i = $running.Count - 1; $i -ge 0; $i--) {
        $entry = $running[$i]
        $proc = $entry.Process
        if ($proc.HasExited) {
            $duration = (Get-Date) - $entry.Started
            Write-LaunchLog ("Finished PID={0} exit={1} mode={2} {3} {4} seed{5} duration={6:n1}min stdout={7} stderr={8}" -f $proc.Id, $proc.ExitCode, $entry.Job.Mode, $entry.Job.Kind, $entry.Job.Level, $entry.Job.Seed, $duration.TotalMinutes, $entry.Stdout, $entry.Stderr)
            $completed.Add([pscustomobject]@{
                Kind = $entry.Job.Kind
                Level = $entry.Job.Level
                Seed = $entry.Job.Seed
                Mode = $entry.Job.Mode
                ExitCode = $proc.ExitCode
                Stdout = $entry.Stdout
                Stderr = $entry.Stderr
            })
            $running.RemoveAt($i)
        }
    }
}

$summaryPath = Join-Path $LaunchLogDir ("level3_belief_only_reboot_resume_summary_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$completed | Export-Csv -NoTypeInformation -Path $summaryPath
$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
    Write-LaunchLog ("Completed with {0} failed jobs. Summary: {1}" -f $failed.Count, $summaryPath)
    exit 1
}

Write-LaunchLog ("All queued incomplete Level 3 belief-only jobs completed. Summary: {0}" -f $summaryPath)
