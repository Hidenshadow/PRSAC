param(
  [int]$Seed = 0,
  [int]$MaxParallel = 3,
  [string]$OutputRoot = "runs/rl_baselines_refresh_1seed",
  [string]$Python = ".\.venv\Scripts\python.exe",
  [int]$NominalTimesteps = 50000,
  [int]$RecoveryTimesteps = 20480,
  [int]$EvalInterval = 1024,
  [int]$NumEvalEpisodes = 128,
  [int]$TrainEvalEpisodes = 64,
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

$outputRootPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot
} else {
  Join-Path $repo $OutputRoot
}
$logsDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null

$cases = @(
  @{ Algo = "ppo"; Level = "level2"; Difficulty = "easy"; LevelConfig = "configs/levels/ppo_difficulty/level2_easy.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" },
  @{ Algo = "sac"; Level = "level2"; Difficulty = "easy"; LevelConfig = "configs/levels/ppo_difficulty/level2_easy.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" },
  @{ Algo = "ppo"; Level = "level2"; Difficulty = "medium"; LevelConfig = "configs/levels/ppo_difficulty/level2_medium.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" },
  @{ Algo = "sac"; Level = "level2"; Difficulty = "medium"; LevelConfig = "configs/levels/ppo_difficulty/level2_medium.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" },
  @{ Algo = "ppo"; Level = "level3"; Difficulty = "medium"; LevelConfig = "configs/levels/ppo_difficulty/level3_medium.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" },
  @{ Algo = "sac"; Level = "level3"; Difficulty = "medium"; LevelConfig = "configs/levels/ppo_difficulty/level3_medium.json"; BaseConfig = "configs/ppo_lunar_viper_relative_reward.json" }
)

$running = @()
$completed = @()
$launchRecords = @()

function Start-RefreshCase {
  param([hashtable]$Case)

  $scenario = "$($Case.Level)_$($Case.Difficulty)"
  $caseRoot = Join-Path $outputRootPath "$($Case.Algo)\${scenario}_shock_recovery_1seed"
  $seedDir = Join-Path $caseRoot "seed$Seed"
  $logPrefix = "refresh_$($Case.Algo)_${scenario}_seed$Seed"
  $stdout = Join-Path $logsDir "$logPrefix.out.log"
  $stderr = Join-Path $logsDir "$logPrefix.err.log"

  $argsList = @(
    "-u", "run_shock_recovery_experiment.py",
    "--algo", $Case.Algo,
    "--level-config", $Case.LevelConfig,
    "--base-config", $Case.BaseConfig,
    "--output-dir", $seedDir,
    "--seed", "$Seed",
    "--nominal-timesteps", "$NominalTimesteps",
    "--recovery-timesteps", "$RecoveryTimesteps",
    "--eval-interval", "$EvalInterval",
    "--num-eval-episodes", "$NumEvalEpisodes",
    "--train-eval-episodes", "$TrainEvalEpisodes",
    "--device", $Device,
    "--clean-output"
  )

  Write-Host "Starting $($Case.Algo) $scenario seed$Seed -> $seedDir"
  $process = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

  $record = [PSCustomObject]@{
    Algo = $Case.Algo
    Level = $Case.Level
    Difficulty = $Case.Difficulty
    Scenario = $scenario
    Seed = $Seed
    Pid = $process.Id
    OutputRoot = $caseRoot
    SeedDir = $seedDir
    Stdout = $stdout
    Stderr = $stderr
    Process = $process
  }
  $script:launchRecords += $record
  return $record
}

$queue = [System.Collections.Queue]::new()
foreach ($case in $cases) {
  $queue.Enqueue($case)
}

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    $running += Start-RefreshCase -Case $queue.Dequeue()
  }

  Start-Sleep -Seconds 10
  $stillRunning = @()
  foreach ($item in $running) {
    $item.Process.Refresh()
    if ($item.Process.HasExited) {
      $exitCode = $item.Process.ExitCode
      Write-Host "Finished $($item.Algo) $($item.Scenario) seed$Seed exit=$exitCode"
      $completed += [PSCustomObject]@{
        Algo = $item.Algo
        Scenario = $item.Scenario
        Seed = $Seed
        ExitCode = $exitCode
        OutputRoot = $item.OutputRoot
        Stdout = $item.Stdout
        Stderr = $item.Stderr
      }
    } else {
      $stillRunning += $item
    }
  }
  $running = $stillRunning
}

$manifestPath = Join-Path $outputRootPath "refresh_1seed_manifest.json"
$launchRecords |
  Select-Object Algo,Level,Difficulty,Scenario,Seed,Pid,OutputRoot,SeedDir,Stdout,Stderr |
  ConvertTo-Json -Depth 4 |
  Set-Content -Path $manifestPath -Encoding UTF8

$summaryPath = Join-Path $outputRootPath "refresh_1seed_process_summary.csv"
$completed | Export-Csv -Path $summaryPath -NoTypeInformation

$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
  Write-Error "One or more refresh runs failed. See $summaryPath"
}

$caseRoots = $completed | Select-Object -ExpandProperty OutputRoot -Unique
foreach ($caseRoot in $caseRoots) {
  Write-Host "Aggregating $caseRoot"
  & $Python scripts/aggregate_shock_recovery_3seeds.py $caseRoot
}

Write-Host "Done. Manifest: $manifestPath"
