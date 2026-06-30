param(
  [string]$Root = "runs/rl_baselines_refresh_1seed",
  [int]$EvalEpisodes = 300,
  [int]$Seed = 0,
  [int]$MaxParallel = 1,
  [string]$CheckpointMode = "nominal_final",
  [string]$VariantMode = "scale_component",
  [string[]]$Algorithms = @("ppo", "sac"),
  [string[]]$Scenarios = @("level2_easy", "level2_medium", "level3_medium"),
  [switch]$Watch
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

$rootPath = if ([System.IO.Path]::IsPathRooted($Root)) {
  $Root
} else {
  Join-Path $repo $Root
}
$outputRoot = Join-Path $rootPath "attack_suite_eval"
$logsDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$cases = @()
foreach ($algo in $Algorithms) {
  foreach ($scenario in $Scenarios) {
    $seedDir = Join-Path $rootPath "$algo\${scenario}_shock_recovery_1seed\seed$Seed"
    $caseOutput = Join-Path $outputRoot "$algo\${scenario}_seed$Seed"
    $cases += [PSCustomObject]@{
      Algo = $algo
      Scenario = $scenario
      SeedDir = $seedDir
      OutputDir = $caseOutput
    }
  }
}

function Test-CompletedRun {
  param([string]$SeedDir)
  return (
    (Test-Path (Join-Path $SeedDir "shock_recovery_summary.csv")) -and
    (Test-Path (Join-Path $SeedDir "shock_recovery_curve.csv")) -and
    (Test-Path (Join-Path $SeedDir "checkpoints\checkpoint_nominal.pt"))
  )
}

function Start-AttackSuiteCase {
  param([object]$Case)

  $summaryOut = Join-Path $Case.OutputDir "attack_suite_summary.csv"
  if (Test-Path $summaryOut) {
    Write-Host "Skipping existing $($Case.Algo) $($Case.Scenario): $summaryOut"
    return $null
  }

  New-Item -ItemType Directory -Force -Path $Case.OutputDir | Out-Null
  $prefix = "attack_suite_$($Case.Algo)_$($Case.Scenario)_seed$Seed"
  $stdout = Join-Path $logsDir "$prefix.out.log"
  $stderr = Join-Path $logsDir "$prefix.err.log"
  $argsList = @(
    "-u", "scripts\evaluate_attack_suite.py",
    "--source-run-dir", $Case.SeedDir,
    "--output-dir", $Case.OutputDir,
    "--eval-episodes", "$EvalEpisodes",
    "--seed", "$Seed",
    "--checkpoint-mode", $CheckpointMode,
    "--variant-mode", $VariantMode
  )

  Write-Host "Starting attack-suite eval $($Case.Algo) $($Case.Scenario) -> $($Case.OutputDir)"
  $process = Start-Process `
    -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

  return [PSCustomObject]@{
    Algo = $Case.Algo
    Scenario = $Case.Scenario
    OutputDir = $Case.OutputDir
    Stdout = $stdout
    Stderr = $stderr
    Process = $process
  }
}

$pending = [System.Collections.Queue]::new()
foreach ($case in $cases) {
  $pending.Enqueue($case)
}

$running = @()
$completed = @()
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
  $checkedPending = 0
  $initialPending = $pending.Count
  while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel -and $checkedPending -lt $initialPending) {
    $case = $pending.Dequeue()
    if (-not (Test-CompletedRun -SeedDir $case.SeedDir)) {
      if ($Watch) {
        $pending.Enqueue($case)
        $checkedPending += 1
        continue
      }
      Write-Host "Skipping incomplete $($case.Algo) $($case.Scenario): $($case.SeedDir)"
      continue
    }
    $started = Start-AttackSuiteCase -Case $case
    if ($null -ne $started) {
      $running += $started
    }
    $checkedPending += 1
  }

  if ($running.Count -eq 0) {
    if ($pending.Count -gt 0 -and $Watch) {
      $next = $pending.Peek()
      Write-Host "Waiting for completed run: $($next.Algo) $($next.Scenario)"
      Start-Sleep -Seconds 60
      continue
    }
    break
  }

  Start-Sleep -Seconds 10
  $stillRunning = @()
  foreach ($item in $running) {
    $item.Process.Refresh()
    if ($item.Process.HasExited) {
      Write-Host "Finished attack-suite eval $($item.Algo) $($item.Scenario) exit=$($item.Process.ExitCode)"
      $completed += [PSCustomObject]@{
        Algo = $item.Algo
        Scenario = $item.Scenario
        ExitCode = $item.Process.ExitCode
        OutputDir = $item.OutputDir
        Stdout = $item.Stdout
        Stderr = $item.Stderr
      }
    } else {
      $stillRunning += $item
    }
  }
  $running = $stillRunning
}

$summaryPath = Join-Path $outputRoot "attack_suite_process_summary.csv"
$completed | Export-Csv -Path $summaryPath -NoTypeInformation

$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
  Write-Error "One or more attack-suite evaluations failed. See $summaryPath"
}

Write-Host "Done. Process summary: $summaryPath"
