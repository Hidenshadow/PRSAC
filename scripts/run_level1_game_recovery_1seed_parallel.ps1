param(
    [int]$Seed = 0,
    [string]$OutputBase = "runs/game_recovery_level1",
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

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ResolvedOutputBase = Join-Path $Root $OutputBase
New-Item -ItemType Directory -Force -Path $ResolvedOutputBase | Out-Null

$Difficulties = @("easy", "medium", "hard")
$Launches = @()
$RunningProcesses = @()

foreach ($Difficulty in $Difficulties) {
    $RunRoot = Join-Path $ResolvedOutputBase "level1_${Difficulty}_shock_recovery_1seed"
    $OutputDir = Join-Path $RunRoot "seed$Seed"
    $Stdout = Join-Path $LogDir "level1_${Difficulty}_game_recovery_seed${Seed}.out.log"
    $Stderr = Join-Path $LogDir "level1_${Difficulty}_game_recovery_seed${Seed}.err.log"

    $Arguments = @(
        "-u",
        "run_shock_recovery_experiment.py",
        "--algo", "ppo",
        "--level-config", "configs\levels\ppo_difficulty\level1_${Difficulty}.json",
        "--output-dir", $OutputDir,
        "--seed", [string]$Seed,
        "--nominal-timesteps", [string]$NominalTimesteps,
        "--recovery-timesteps", [string]$RecoveryTimesteps,
        "--eval-interval", [string]$EvalInterval,
        "--num-eval-episodes", [string]$NumEvalEpisodes,
        "--train-eval-episodes", [string]$TrainEvalEpisodes,
        "--in-domain-seed", "909",
        "--heldout-seed", "1919",
        "--device", "cpu",
        "--game-recovery-enabled"
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

    $Launches += [pscustomobject]@{
        difficulty = $Difficulty
        seed = $Seed
        pid = $Process.Id
        output_dir = $OutputDir
        stdout = $Stdout
        stderr = $Stderr
    }
    $RunningProcesses += [pscustomobject]@{
        difficulty = $Difficulty
        process = $Process
        stdout = $Stdout
        stderr = $Stderr
    }
}

$ManifestPath = Join-Path $ResolvedOutputBase "level1_game_recovery_1seed_parallel_launch.json"
$Launches | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8

$Launches | Format-Table -AutoSize
Write-Host "Launch manifest: $ManifestPath"

if ($Wait) {
    $Failed = $false
    foreach ($Run in $RunningProcesses) {
        $Run.process.WaitForExit()
        $ExitCode = $Run.process.ExitCode
        if ([int]$ExitCode -ne 0) {
            $Failed = $true
            Write-Error "Run failed: difficulty=$($Run.difficulty) pid=$($Run.process.Id) exit_code=$ExitCode stderr=$($Run.stderr)"
        }
    }
    if ($Failed) {
        exit 1
    }
}
