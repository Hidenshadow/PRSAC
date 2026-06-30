param(
  [string]$OutputRoot = "runs/rl_baselines",
  [int[]]$Seeds = @(0, 1, 2, 3, 4),
  [int]$MaxParallel = 4,
  [int]$TargetCpuPercent = 90,
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
New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$logicalProcessors = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$requestedAffinityProcessors = [Math]::Max(1, [Math]::Floor($logicalProcessors * ($TargetCpuPercent / 100.0)))
$affinityProcessors = if ($TargetCpuPercent -lt 100 -and $requestedAffinityProcessors -gt 1) {
  $requestedAffinityProcessors - 1
} else {
  $requestedAffinityProcessors
}
if ($affinityProcessors -ge 63) {
  $processAffinityMask = [IntPtr]::new(-1)
} else {
  $processAffinityMask = [IntPtr]::new(([int64]1 -shl $affinityProcessors) - 1)
}
Write-Host "CPU affinity target: $affinityProcessors / $logicalProcessors logical processors."

$cases = @(
  @{ Algo = "ppo"; Level = "level2"; Difficulty = "easy"; Scenario = "level2_easy"; LevelConfig = "configs/levels/ppo_difficulty/level2_easy.json" },
  @{ Algo = "ppo"; Level = "level2"; Difficulty = "medium"; Scenario = "level2_medium"; LevelConfig = "configs/levels/ppo_difficulty/level2_medium.json" },
  @{ Algo = "ppo"; Level = "level3"; Difficulty = "medium"; Scenario = "level3_medium"; LevelConfig = "configs/levels/ppo_difficulty/level3_medium.json" },
  @{ Algo = "sac"; Level = "level2"; Difficulty = "easy"; Scenario = "level2_easy"; LevelConfig = "configs/levels/ppo_difficulty/level2_easy.json" },
  @{ Algo = "sac"; Level = "level2"; Difficulty = "medium"; Scenario = "level2_medium"; LevelConfig = "configs/levels/ppo_difficulty/level2_medium.json" },
  @{ Algo = "sac"; Level = "level3"; Difficulty = "medium"; Scenario = "level3_medium"; LevelConfig = "configs/levels/ppo_difficulty/level3_medium.json" }
)

function Get-CaseRoot {
  param([hashtable]$Case)
  return Join-Path $outputRootPath "$($Case.Algo)\$($Case.Scenario)_shock_recovery_5seeds"
}

function Test-SeedComplete {
  param([string]$SeedDir)
  return (
    (Test-Path (Join-Path $SeedDir "shock_recovery_summary.csv")) -and
    (Test-Path (Join-Path $SeedDir "shock_recovery_curve.csv")) -and
    (Test-Path (Join-Path $SeedDir "checkpoints\checkpoint_nominal.pt"))
  )
}

function Test-SeedRunning {
  param([string]$SeedDir)
  $escapedSeedDir = [regex]::Escape($SeedDir)
  $processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match $escapedSeedDir)
  })
  return ($processes.Count -gt 0)
}

function Get-TotalCpuPercent {
  try {
    $sample = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1
    return [double]$sample.CounterSamples[0].CookedValue
  } catch {
    return 0.0
  }
}

function Start-BaselineSeed {
  param([hashtable]$Case, [int]$Seed)

  $caseRoot = Get-CaseRoot -Case $Case
  $seedDir = Join-Path $caseRoot "seed$Seed"
  New-Item -ItemType Directory -Force -Path $caseRoot | Out-Null

  $logPrefix = "missing_rl_baseline_$($Case.Algo)_$($Case.Scenario)_seed$Seed"
  $stdout = Join-Path $logsDir "$logPrefix.out.log"
  $stderr = Join-Path $logsDir "$logPrefix.err.log"
  $argsList = @(
    "-u", "run_shock_recovery_experiment.py",
    "--algo", $Case.Algo,
    "--level-config", $Case.LevelConfig,
    "--base-config", "configs/ppo_lunar_viper_relative_reward.json",
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

  Write-Host "Starting $($Case.Algo) $($Case.Scenario) seed$Seed -> $seedDir"
  $process = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
  $process.ProcessorAffinity = $processAffinityMask

  return [PSCustomObject]@{
    Algo = $Case.Algo
    Scenario = $Case.Scenario
    Seed = $Seed
    OutputRoot = $caseRoot
    SeedDir = $seedDir
    Stdout = $stdout
    Stderr = $stderr
    Process = $process
  }
}

$queue = [System.Collections.Queue]::new()
foreach ($case in $cases) {
  foreach ($seed in $Seeds) {
    $seedDir = Join-Path (Get-CaseRoot -Case $case) "seed$seed"
    if (Test-SeedComplete -SeedDir $seedDir) {
      Write-Host "Skipping complete $($case.Algo) $($case.Scenario) seed$seed"
      continue
    }
    if (Test-SeedRunning -SeedDir $seedDir) {
      Write-Host "Skipping running $($case.Algo) $($case.Scenario) seed$seed"
      continue
    }
    $queue.Enqueue([PSCustomObject]@{ Case = $case; Seed = $seed })
  }
}

$running = @()
$completed = @()
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    $cpu = Get-TotalCpuPercent
    if ($cpu -ge $TargetCpuPercent) {
      Write-Host ("CPU {0:N1}% >= target {1}%; waiting before launching more." -f $cpu, $TargetCpuPercent)
      break
    }
    $item = $queue.Dequeue()
    $running += Start-BaselineSeed -Case $item.Case -Seed ([int]$item.Seed)
    Start-Sleep -Seconds 15
  }

  if ($running.Count -eq 0 -and $queue.Count -eq 0) {
    break
  }

  Start-Sleep -Seconds 15
  $stillRunning = @()
  foreach ($item in $running) {
    $item.Process.Refresh()
    if ($item.Process.HasExited) {
      Write-Host "Finished $($item.Algo) $($item.Scenario) seed$($item.Seed) exit=$($item.Process.ExitCode)"
      $completed += [PSCustomObject]@{
        Algo = $item.Algo
        Scenario = $item.Scenario
        Seed = $item.Seed
        ExitCode = $item.Process.ExitCode
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

$summaryPath = Join-Path $outputRootPath "missing_rl_baseline_seed_process_summary.csv"
$completed | Export-Csv -Path $summaryPath -NoTypeInformation
$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
  Write-Error "One or more missing baseline seed runs failed. See $summaryPath"
}

$caseRoots = @()
foreach ($case in $cases) {
  $caseRoot = Get-CaseRoot -Case $case
  if (Test-Path $caseRoot) {
    $caseRoots += $caseRoot
  }
}

foreach ($caseRoot in ($caseRoots | Select-Object -Unique)) {
  $curves = @(Get-ChildItem $caseRoot -Recurse -Filter shock_recovery_curve.csv -ErrorAction SilentlyContinue)
  if ($curves.Count -gt 0) {
    Write-Host "Aggregating $caseRoot"
    & $Python scripts\aggregate_shock_recovery_3seeds.py $caseRoot
  }
}

Write-Host "Done. Process summary: $summaryPath"
