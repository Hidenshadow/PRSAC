param(
    [int[]]$Seeds = @(0),
    [string]$OutputName = "ppo_ldac_level1_easy_seed0_20260623",
    [int]$TorchThreadsPerProcess = 2,
    [int]$NominalTimesteps = 50000,
    [int]$RecoveryTimesteps = 20480,
    [int]$EvalInterval = 1024,
    [int]$NumEvalEpisodes = 300,
    [int]$TrainEvalEpisodes = 64,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runner = Join-Path $Root "run_shock_recovery_experiment.py"
$Config = Join-Path $Root "configs\levels\ppo_difficulty\level1_easy.json"
$OutRoot = Join-Path $Root (Join-Path "runs\ppo_modified" $OutputName)
$LogRoot = Join-Path $OutRoot "logs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing Python executable: $Python"
}
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Missing runner: $Runner"
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Missing level config: $Config"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:MKL_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:OPENBLAS_NUM_THREADS = [string]$TorchThreadsPerProcess
$env:NUMEXPR_NUM_THREADS = [string]$TorchThreadsPerProcess

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$running = @()
foreach ($seed in $Seeds) {
    $JobOut = Join-Path $OutRoot (Join-Path "level1_easy" ("seed" + $seed))
    $LogPrefix = "level1_easy_seed{0}" -f $seed
    $Stdout = Join-Path $LogRoot ($LogPrefix + "_stdout.log")
    $Stderr = Join-Path $LogRoot ($LogPrefix + "_stderr.log")

    $ArgsList = @(
        $Runner,
        "--algo", "ppo",
        "--level-config", $Config,
        "--output-dir", $JobOut,
        "--nominal-timesteps", [string]$NominalTimesteps,
        "--recovery-timesteps", [string]$RecoveryTimesteps,
        "--eval-interval", [string]$EvalInterval,
        "--num-eval-episodes", [string]$NumEvalEpisodes,
        "--train-eval-episodes", [string]$TrainEvalEpisodes,
        "--seed", [string]$seed,
        "--in-domain-seed", "909",
        "--heldout-seed", "1919",
        "--python", $Python,
        "--device", "cpu",
        "--clean-output",
        "--game-recovery-enabled",
        "--game-attack-sampler", "adaptive_bandit",
        "--game-attack-mixture-size", "5",
        "--game-attack-jitter", "0.15",
        "--game-attack-variant-mode", "scale",
        "--game-nominal-prior-coef", "0.25",
        "--game-lambda-drift-coef", "0.0",
        "--game-risk-drift-coef", "0.0"
    )
    if ($DryRun) {
        $ArgsList += "--dry-run"
    }

    $proc = Start-Process `
        -FilePath $Python `
        -ArgumentList $ArgsList `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr

    $running += [pscustomobject]@{
        Process = $proc
        Level = "level1_easy"
        Seed = $seed
        Output = $JobOut
        Stdout = $Stdout
        Stderr = $Stderr
    }
    Write-Host ("Started PPO-LDAC {0} seed{1}: PID={2}" -f "level1_easy", $seed, $proc.Id)
}

$ManifestPath = Join-Path $OutRoot "launch_manifest.json"
$running | ForEach-Object {
    [pscustomobject]@{
        level = $_.Level
        seed = $_.Seed
        pid = $_.Process.Id
        output = $_.Output
        stdout = $_.Stdout
        stderr = $_.Stderr
        nominal_timesteps = $NominalTimesteps
        recovery_timesteps = $RecoveryTimesteps
        eval_interval = $EvalInterval
        num_eval_episodes = $NumEvalEpisodes
        train_eval_episodes = $TrainEvalEpisodes
        game_nominal_prior_coef = 0.25
        game_attack_sampler = "adaptive_bandit"
    }
} | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8

$failed = $false
while ($running.Count -gt 0) {
    Start-Sleep -Seconds 30
    $stillRunning = @()
    foreach ($item in $running) {
        if ($item.Process.HasExited) {
            $exitCode = [int]$item.Process.ExitCode
            Write-Host ("Finished PPO-LDAC {0} seed{1}: exit={2}" -f $item.Level, $item.Seed, $exitCode)
            if ($exitCode -ne 0) {
                $failed = $true
                Write-Host ("stderr: {0}" -f $item.Stderr)
            }
        } else {
            $stillRunning += $item
        }
    }
    $running = $stillRunning
}

if ($failed) {
    exit 1
}

Write-Host ("All PPO-LDAC level1_easy experiments finished: {0}" -f $OutRoot)
