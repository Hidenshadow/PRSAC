param(
    [int[]]$Seeds = @(0, 1, 2),
    [string[]]$Levels = @("level1", "level2", "level3"),
    [string[]]$Difficulties = @("easy", "medium", "hard"),
    [int]$MaxParallel = 9,
    [string]$OutputBase = "runs/ap_cvar_game_recovery_matrix",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$NominalTimesteps = 50000,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [int]$TrainEvalEpisodes = 64,
    [int]$TorchThreadsPerProcess = 1,
    [double]$GameBanditEta = 0.8,
    [double]$GameBanditMinProb = 0.05,
    [double]$GameBanditPriorMix = 0.25,
    [double]$BenchmarkFloor = 0.30,
    [ValidateSet("scale", "component", "scale_component")]
    [string]$AttackVariantMode = "component",
    [double]$CvarQuantile = 0.75,
    [double]$CvarWeight = 1.5,
    [double]$TailExcessWeight = 0.75,
    [double]$RiskFeatureWeight = 0.25,
    [double]$WeightCap = 4.0,
    [bool]$CleanOutput = $true,
    [switch]$DryRun,
    [int]$PollSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1) {
    throw "MaxParallel must be positive."
}

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (Test-Path -LiteralPath $Python) {
    $PythonExe = (Resolve-Path -LiteralPath $Python).Path
} else {
    $PythonExe = $Python
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:MKL_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:OPENBLAS_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:NUMEXPR_NUM_THREADS = [string]$TorchThreadsPerProcess

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ResolvedOutputBase = Join-Path $Root $OutputBase
New-Item -ItemType Directory -Force -Path $ResolvedOutputBase | Out-Null

$ManifestPath = Join-Path $ResolvedOutputBase "ap_cvar_matrix_launch.json"
$Pending = [System.Collections.Queue]::new()
$AllLaunches = [System.Collections.Generic.List[object]]::new()
$Running = @()

foreach ($Seed in $Seeds) {
    foreach ($Level in $Levels) {
        foreach ($Difficulty in $Difficulties) {
            $LevelConfig = "configs\levels\ppo_difficulty\${Level}_${Difficulty}.json"
            if (-not (Test-Path -LiteralPath $LevelConfig)) {
                throw "Missing level config: $LevelConfig"
            }
            $Pending.Enqueue([pscustomobject]@{
                seed = [int]$Seed
                level = [string]$Level
                difficulty = [string]$Difficulty
                level_config = $LevelConfig
            })
        }
    }
}

function Write-LaunchManifest {
    $AllLaunches | ConvertTo-Json -Depth 5 | Set-Content -Path $ManifestPath -Encoding UTF8
}

function Start-Experiment {
    param([pscustomobject]$Case)

    $RunRoot = Join-Path $ResolvedOutputBase "$($Case.level)_$($Case.difficulty)_shock_recovery_1seed"
    $OutputDir = Join-Path $RunRoot "seed$($Case.seed)"
    $Stdout = Join-Path $LogDir "$($Case.level)_$($Case.difficulty)_ap_cvar_seed$($Case.seed).out.log"
    $Stderr = Join-Path $LogDir "$($Case.level)_$($Case.difficulty)_ap_cvar_seed$($Case.seed).err.log"

    $Arguments = @(
        "-u",
        "run_shock_recovery_experiment.py",
        "--algo", "ppo",
        "--level-config", $Case.level_config,
        "--output-dir", $OutputDir,
        "--seed", [string]$Case.seed,
        "--nominal-timesteps", [string]$NominalTimesteps,
        "--recovery-timesteps", [string]$RecoveryTimesteps,
        "--eval-interval", [string]$EvalInterval,
        "--num-eval-episodes", [string]$NumEvalEpisodes,
        "--train-eval-episodes", [string]$TrainEvalEpisodes,
        "--device", "cpu",
        "--game-recovery-enabled",
        "--game-attack-sampler", "adaptive_bandit",
        "--game-attack-variant-mode", $AttackVariantMode,
        "--game-bandit-eta", [string]$GameBanditEta,
        "--game-bandit-min-prob", [string]$GameBanditMinProb,
        "--game-bandit-prior-mix", [string]$GameBanditPriorMix,
        "--game-bandit-benchmark-floor", [string]$BenchmarkFloor,
        "--game-nominal-prior-coef", "0.0",
        "--game-lambda-drift-coef", "0.0",
        "--game-risk-drift-coef", "0.0",
        "--ap-cvar-enabled",
        "--ap-cvar-quantile", [string]$CvarQuantile,
        "--ap-cvar-weight", [string]$CvarWeight,
        "--ap-cvar-tail-excess-weight", [string]$TailExcessWeight,
        "--ap-cvar-risk-feature-weight", [string]$RiskFeatureWeight,
        "--ap-cvar-weight-cap", [string]$WeightCap
    )
    if ($CleanOutput) {
        $Arguments += "--clean-output"
    }
    if ($DryRun) {
        $Arguments += "--dry-run"
    }

    $Process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -WindowStyle Hidden `
        -PassThru

    $Launch = [pscustomobject]@{
        level = $Case.level
        difficulty = $Case.difficulty
        seed = $Case.seed
        pid = $Process.Id
        status = "running"
        exit_code = $null
        started_at = (Get-Date).ToString("s")
        completed_at = $null
        output_dir = $OutputDir
        stdout = $Stdout
        stderr = $Stderr
    }
    $AllLaunches.Add($Launch)
    Write-LaunchManifest
    Write-Host "Started $($Case.level)/$($Case.difficulty) seed=$($Case.seed) pid=$($Process.Id)"
    return [pscustomobject]@{
        launch = $Launch
        process = $Process
    }
}

Write-Host "AP-CVaR matrix queued: $($Pending.Count) runs; max_parallel=$MaxParallel"
while ($Pending.Count -gt 0 -or $Running.Count -gt 0) {
    while ($Pending.Count -gt 0 -and $Running.Count -lt $MaxParallel) {
        $Running += Start-Experiment -Case $Pending.Dequeue()
    }

    if ($Running.Count -eq 0) {
        break
    }

    Start-Sleep -Seconds ([Math]::Max($PollSeconds, 5))
    $StillRunning = @()
    foreach ($Run in $Running) {
        if ($Run.process.HasExited) {
            $Run.launch.exit_code = [int]$Run.process.ExitCode
            $Run.launch.completed_at = (Get-Date).ToString("s")
            if ([int]$Run.process.ExitCode -eq 0) {
                $Run.launch.status = "completed"
                Write-Host "Completed $($Run.launch.level)/$($Run.launch.difficulty) seed=$($Run.launch.seed)"
            } else {
                $Run.launch.status = "failed"
                Write-Host "Failed $($Run.launch.level)/$($Run.launch.difficulty) seed=$($Run.launch.seed) exit=$($Run.process.ExitCode)"
            }
        } else {
            $StillRunning += $Run
        }
    }
    $Running = $StillRunning
    Write-LaunchManifest
}

$Failed = @($AllLaunches | Where-Object { $_.status -eq "failed" })
Write-Host "AP-CVaR matrix finished. completed=$(@($AllLaunches | Where-Object { $_.status -eq 'completed' }).Count) failed=$($Failed.Count)"
Write-Host "Launch manifest: $ManifestPath"
if ($Failed.Count -gt 0) {
    exit 1
}
