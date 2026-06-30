param(
    [int]$Seed = 0,
    [string[]]$Levels = @("level2", "level3"),
    [string]$OutputBase = "runs/game_recovery_chain",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$NominalTimesteps = 50000,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [int]$TrainEvalEpisodes = 64,
    [int]$TorchThreadsPerProcess = 1,
    [bool]$CleanOutput = $true,
    [switch]$DryRun
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

$Difficulties = @("easy", "medium", "hard")
$ResolvedLevels = @()
foreach ($LevelText in $Levels) {
    foreach ($Token in ([string]$LevelText).Split(",")) {
        $LevelName = $Token.Trim()
        if ($LevelName.Length -gt 0) {
            $ResolvedLevels += $LevelName
        }
    }
}
if ($ResolvedLevels.Count -eq 0) {
    throw "No levels requested."
}
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ResolvedOutputBase = Join-Path $Root $OutputBase
New-Item -ItemType Directory -Force -Path $ResolvedOutputBase | Out-Null

$AllLaunches = @()
$ChainFailed = $false

foreach ($Level in $ResolvedLevels) {
    Write-Host "=== Starting $Level game-recovery seed=$Seed ==="
    $LevelProcesses = @()
    foreach ($Difficulty in $Difficulties) {
        $LevelConfig = "configs\levels\ppo_difficulty\${Level}_${Difficulty}.json"
        if (-not (Test-Path -LiteralPath $LevelConfig)) {
            throw "Missing level config: $LevelConfig"
        }

        $RunRoot = Join-Path $ResolvedOutputBase "${Level}_${Difficulty}_shock_recovery_1seed"
        $OutputDir = Join-Path $RunRoot "seed$Seed"
        $Stdout = Join-Path $LogDir "${Level}_${Difficulty}_game_recovery_seed${Seed}.out.log"
        $Stderr = Join-Path $LogDir "${Level}_${Difficulty}_game_recovery_seed${Seed}.err.log"

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

        $Launch = [pscustomobject]@{
            level = $Level
            difficulty = $Difficulty
            seed = $Seed
            pid = $Process.Id
            output_dir = $OutputDir
            stdout = $Stdout
            stderr = $Stderr
        }
        $AllLaunches += $Launch
        $LevelProcesses += [pscustomobject]@{
            launch = $Launch
            process = $Process
        }
        Write-Host "Started $Level/$Difficulty seed=$Seed pid=$($Process.Id)"
    }

    $ManifestPath = Join-Path $ResolvedOutputBase "game_recovery_level_chain_seed${Seed}_launch.json"
    $AllLaunches | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8
    Write-Host "Launch manifest: $ManifestPath"

    foreach ($Run in $LevelProcesses) {
        $Run.process.WaitForExit()
        $ExitCode = [int]$Run.process.ExitCode
        if ($ExitCode -ne 0) {
            $ChainFailed = $true
            Write-Error "Run failed: level=$($Run.launch.level) difficulty=$($Run.launch.difficulty) pid=$($Run.process.Id) exit_code=$ExitCode stderr=$($Run.launch.stderr)"
        } else {
            Write-Host "Completed $($Run.launch.level)/$($Run.launch.difficulty) seed=$Seed"
        }
    }

    if ($ChainFailed) {
        Write-Error "Stopping chain after $Level because at least one run failed."
        exit 1
    }
    Write-Host "=== Completed $Level game-recovery seed=$Seed ==="
}

Write-Host "All requested levels complete."
