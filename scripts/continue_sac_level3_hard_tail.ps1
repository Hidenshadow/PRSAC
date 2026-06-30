param(
  [string]$OutputRoot = "runs/rl_baselines/sac/level3_hard_shock_recovery_5seeds",
  [int[]]$Seeds = @(0, 1, 2, 3),
  [int]$MaxParallel = 4,
  [int]$TargetCpuPercent = 80,
  [string]$Python = ".\.venv\Scripts\python.exe",
  [int]$TargetStep = 20480,
  [string]$Device = "cpu",
  [switch]$SkipRebuildCurve
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

$logicalProcessors = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$requestedAffinityProcessors = [Math]::Max(1, [Math]::Floor($logicalProcessors * ($TargetCpuPercent / 100.0)))
$affinityProcessors = if ($TargetCpuPercent -lt 100 -and $requestedAffinityProcessors -gt 1) {
  $requestedAffinityProcessors - 1
} else {
  $requestedAffinityProcessors
}
$processAffinityMask = if ($affinityProcessors -ge 63) {
  [IntPtr]::new(-1)
} else {
  [IntPtr]::new(([int64]1 -shl $affinityProcessors) - 1)
}
Write-Host "CPU affinity target: $affinityProcessors / $logicalProcessors logical processors."

function Get-TotalCpuPercent {
  try {
    $sample = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1
    return [double]$sample.CounterSamples[0].CookedValue
  } catch {
    return 0.0
  }
}

function Test-SeedComplete {
  param([string]$SeedDir)
  return (
    (Test-Path (Join-Path $SeedDir "shock_recovery_summary.csv")) -and
    (Test-Path (Join-Path $SeedDir "shock_recovery_curve.csv")) -and
    (Test-Path (Join-Path $SeedDir "checkpoints\checkpoint_recovery_step_$TargetStep.pt"))
  )
}

function Start-SeedTail {
  param([int]$Seed)

  $seedDir = Join-Path $outputRootPath "seed$Seed"
  if (-not (Test-Path (Join-Path $seedDir "run_config.json"))) {
    throw "Missing run_config.json for seed$Seed at $seedDir"
  }

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $logPrefix = "sac_level3_hard_tail_seed$Seed`_$stamp"
  $stdout = Join-Path $logsDir "$logPrefix.out.log"
  $stderr = Join-Path $logsDir "$logPrefix.err.log"
  $argsList = @(
    "-u", "scripts\continue_shock_recovery_tail.py",
    "--seed-dir", $seedDir,
    "--target-step", "$TargetStep",
    "--python", $Python,
    "--device", $Device
  )
  if ($SkipRebuildCurve) {
    $argsList += "--skip-rebuild-curve"
  }

  Write-Host "Continuing SAC level3_hard seed$Seed -> $seedDir"
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
    Seed = $Seed
    SeedDir = $seedDir
    Stdout = $stdout
    Stderr = $stderr
    Process = $process
  }
}

$queue = [System.Collections.Queue]::new()
foreach ($seed in $Seeds) {
  $seedDir = Join-Path $outputRootPath "seed$seed"
  if (Test-SeedComplete -SeedDir $seedDir) {
    Write-Host "Skipping complete seed$seed."
    continue
  }
  $queue.Enqueue([int]$seed)
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
    $seed = [int]$queue.Dequeue()
    $running += Start-SeedTail -Seed $seed
    Start-Sleep -Seconds 10
  }

  Start-Sleep -Seconds 15
  $stillRunning = @()
  foreach ($item in $running) {
    $item.Process.Refresh()
    if ($item.Process.HasExited) {
      $item.Process.WaitForExit()
      $exitCode = $item.Process.ExitCode
      Write-Host "Finished SAC level3_hard seed$($item.Seed) exit=$exitCode"
      $completed += [PSCustomObject]@{
        Seed = $item.Seed
        ExitCode = $exitCode
        SeedDir = $item.SeedDir
        Stdout = $item.Stdout
        Stderr = $item.Stderr
      }
    } else {
      $stillRunning += $item
    }
  }
  $running = $stillRunning
}

$summaryPath = Join-Path $logsDir ("sac_level3_hard_tail_process_summary_{0}.csv" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$completed | Export-Csv -Path $summaryPath -NoTypeInformation
$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
  Write-Error "One or more SAC level3_hard tail runs failed. See $summaryPath"
}

Write-Host "Aggregating $outputRootPath"
& $Python scripts\aggregate_shock_recovery_3seeds.py $outputRootPath

Write-Host "Done. Process summary: $summaryPath"
