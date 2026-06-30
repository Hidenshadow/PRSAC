param(
    [int]$MaxParallel = 3,
    [string]$Python = "python",
    [string]$Device = "cpu",
    [switch]$SkipPpoSac,
    [switch]$SkipPrSac
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $Root "run_shock_recovery_experiment.py"
$BaseConfig = Join-Path $Root "configs/ppo_lunar_viper_relative_reward.json"
$LaunchLogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LaunchLogDir | Out-Null
$LaunchLog = Join-Path $LaunchLogDir ("level3_belief_only_rebaseline_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-LaunchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $LaunchLog -Value $line
}

function New-Level3Job {
    param(
        [string]$Kind,
        [string]$Algo,
        [string]$Level,
        [string]$Config,
        [int]$Seed,
        [string]$Output,
        [int]$EvalEpisodes = 128,
        [switch]$PrSac
    )
    [pscustomobject]@{
        Kind = $Kind
        Algo = $Algo
        Level = $Level
        Config = $Config
        Seed = $Seed
        Output = $Output
        EvalEpisodes = $EvalEpisodes
        PrSac = [bool]$PrSac
    }
}

$levelConfigs = @(
    @{ Level = "level3_easy"; Config = "configs/levels/ppo_difficulty/level3_easy.json" },
    @{ Level = "level3_medium"; Config = "configs/levels/ppo_difficulty/level3_medium.json" },
    @{ Level = "level3_hard"; Config = "configs/levels/ppo_difficulty/level3_hard.json" }
)

$jobs = New-Object System.Collections.Generic.List[object]

if (-not $SkipPpoSac) {
    foreach ($algo in @("ppo", "sac")) {
        foreach ($level in $levelConfigs) {
            foreach ($seed in 0..4) {
                $outRoot = Join-Path $Root ("runs/rl_baselines/{0}/{1}_shock_recovery_5seeds" -f $algo, $level.Level)
                $output = Join-Path $outRoot ("seed{0}" -f $seed)
                $jobs.Add((New-Level3Job -Kind "rl_baseline" -Algo $algo -Level $level.Level -Config $level.Config -Seed $seed -Output $output))
            }
        }
    }
}

if (-not $SkipPrSac) {
    $ldacRoots = @{
        "level3_easy:0" = "runs/sac_modified/ldac_easy_seed0_20260621"
        "level3_easy:1" = "runs/sac_modified/ldac_v2_easy_seed1_2_20260621"
        "level3_easy:2" = "runs/sac_modified/ldac_v2_easy_seed1_2_20260621"
        "level3_easy:3" = "runs/sac_modified/ldac_v2_easy_seed3_4_20260623"
        "level3_easy:4" = "runs/sac_modified/ldac_v2_easy_seed3_4_20260623"
        "level3_medium:0" = "runs/sac_modified/ldac_v2_medium_seed0_1_20260621"
        "level3_medium:1" = "runs/sac_modified/ldac_v2_medium_seed0_1_20260621"
        "level3_medium:2" = "runs/sac_modified/ldac_v2_medium_seed2_4_20260623"
        "level3_medium:3" = "runs/sac_modified/ldac_v2_medium_seed2_4_20260623"
        "level3_medium:4" = "runs/sac_modified/ldac_v2_medium_seed2_4_20260623"
        "level3_hard:0" = "runs/sac_modified/ldac_v2_hard_seed0_1_20260621"
        "level3_hard:1" = "runs/sac_modified/ldac_v2_hard_seed0_1_20260621"
        "level3_hard:2" = "runs/sac_modified/ldac_v2_hard_seed2_4_20260623"
        "level3_hard:3" = "runs/sac_modified/ldac_v2_hard_seed2_4_20260623"
        "level3_hard:4" = "runs/sac_modified/ldac_v2_hard_seed2_4_20260623"
    }
    foreach ($level in $levelConfigs) {
        foreach ($seed in 0..4) {
            $rootKey = "{0}:{1}" -f $level.Level, $seed
            $outRoot = Join-Path $Root $ldacRoots[$rootKey]
            $output = Join-Path (Join-Path $outRoot $level.Level) ("seed{0}" -f $seed)
            $jobs.Add((New-Level3Job -Kind "pr_sac" -Algo "sac" -Level $level.Level -Config $level.Config -Seed $seed -Output $output -PrSac))
        }
    }
}

$queue = New-Object System.Collections.Queue
foreach ($job in $jobs) {
    $queue.Enqueue($job)
}

$running = New-Object System.Collections.Generic.List[object]
Write-LaunchLog ("Queued {0} Level 3 belief-only jobs with MaxParallel={1}" -f $jobs.Count, $MaxParallel)

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $job = $queue.Dequeue()
        $outputRoot = Split-Path -Parent $job.Output
        $logRoot = Join-Path $outputRoot "logs"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $logPrefix = "{0}_{1}_seed{2}" -f $job.Kind, $job.Level, $job.Seed
        $stdout = Join-Path $logRoot ($logPrefix + "_stdout.log")
        $stderr = Join-Path $logRoot ($logPrefix + "_stderr.log")

        $argsList = @(
            $Runner,
            "--algo", $job.Algo,
            "--level-config", (Join-Path $Root $job.Config),
            "--base-config", $BaseConfig,
            "--output-dir", $job.Output,
            "--nominal-timesteps", "50000",
            "--recovery-timesteps", "20480",
            "--eval-interval", "1024",
            "--num-eval-episodes", [string]$job.EvalEpisodes,
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

        $proc = Start-Process -FilePath $Python -ArgumentList $argsList -WorkingDirectory $Root -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $running.Add([pscustomobject]@{
            Process = $proc
            Job = $job
            Stdout = $stdout
            Stderr = $stderr
            Started = Get-Date
        })
        Write-LaunchLog ("Started PID={0} {1} {2} seed{3}" -f $proc.Id, $job.Kind, $job.Level, $job.Seed)
    }

    Start-Sleep -Seconds 20

    for ($i = $running.Count - 1; $i -ge 0; $i--) {
        $entry = $running[$i]
        $proc = $entry.Process
        if ($proc.HasExited) {
            $duration = (Get-Date) - $entry.Started
            Write-LaunchLog ("Finished PID={0} exit={1} {2} {3} seed{4} duration={5:n1}min stdout={6} stderr={7}" -f $proc.Id, $proc.ExitCode, $entry.Job.Kind, $entry.Job.Level, $entry.Job.Seed, $duration.TotalMinutes, $entry.Stdout, $entry.Stderr)
            $running.RemoveAt($i)
        }
    }
}

Write-LaunchLog "All queued Level 3 belief-only jobs completed."
