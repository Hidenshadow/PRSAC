param(
    [int]$Seed = 0,
    [string]$OutputBase = "runs/adaptive_bandit_game_recovery_pilot",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$NominalTimesteps = 50000,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [int]$TrainEvalEpisodes = 64,
    [int]$TorchThreadsPerProcess = 2,
    [bool]$CleanOutput = $true,
    [switch]$DryRun,
    [switch]$Wait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

$Cases = @(
    @{ level = "level1"; difficulty = "medium" },
    @{ level = "level2"; difficulty = "hard" },
    @{ level = "level3"; difficulty = "easy" }
)

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ResolvedOutputBase = Join-Path $Root $OutputBase
New-Item -ItemType Directory -Force -Path $ResolvedOutputBase | Out-Null

$Launches = @()
$RunningProcesses = @()

foreach ($Case in $Cases) {
    $Level = [string]$Case.level
    $Difficulty = [string]$Case.difficulty
    $LevelConfig = "configs\levels\ppo_difficulty\${Level}_${Difficulty}.json"
    if (-not (Test-Path -LiteralPath $LevelConfig)) {
        throw "Missing level config: $LevelConfig"
    }

    $RunRoot = Join-Path $ResolvedOutputBase "${Level}_${Difficulty}_shock_recovery_1seed"
    $OutputDir = Join-Path $RunRoot "seed$Seed"
    $Stdout = Join-Path $LogDir "${Level}_${Difficulty}_adaptive_bandit_seed${Seed}.out.log"
    $Stderr = Join-Path $LogDir "${Level}_${Difficulty}_adaptive_bandit_seed${Seed}.err.log"

    $Arguments = @(
        "-u",
        "run_shock_recovery_experiment.py",
        "--algo", "ppo",
        "--level-config", $LevelConfig,
        "--output-dir", $OutputDir,
        "--seed", [string]$Seed,
        "--nominal-timesteps", [string]$NominalTimesteps,
        "--recovery-timesteps", [string]$RecoveryTimesteps,
        "--eval-interval", [string]$EvalInterval,
        "--num-eval-episodes", [string]$NumEvalEpisodes,
        "--train-eval-episodes", [string]$TrainEvalEpisodes,
        "--device", "cpu",
        "--game-recovery-enabled",
        "--game-attack-sampler", "adaptive_bandit",
        "--game-nominal-prior-coef", "0.0",
        "--game-lambda-drift-coef", "0.0",
        "--game-risk-drift-coef", "0.0"
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
        level = $Level
        difficulty = $Difficulty
        seed = $Seed
        pid = $Process.Id
        output_dir = $OutputDir
        stdout = $Stdout
        stderr = $Stderr
    }
    $Launches += $Launch
    $RunningProcesses += [pscustomobject]@{
        launch = $Launch
        process = $Process
    }
}

$ManifestPath = Join-Path $ResolvedOutputBase "adaptive_bandit_pilot_seed${Seed}_launch.json"
$Launches | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8

$Launches | Format-Table -AutoSize
Write-Host "Launch manifest: $ManifestPath"

if ($Wait) {
    $Failed = $false
    foreach ($Run in $RunningProcesses) {
        $Run.process.WaitForExit()
        $ExitCode = [int]$Run.process.ExitCode
        if ($ExitCode -ne 0) {
            $Failed = $true
            Write-Error "Run failed: level=$($Run.launch.level) difficulty=$($Run.launch.difficulty) pid=$($Run.process.Id) exit_code=$ExitCode stderr=$($Run.launch.stderr)"
        }
    }
    if ($Failed) {
        exit 1
    }
}
