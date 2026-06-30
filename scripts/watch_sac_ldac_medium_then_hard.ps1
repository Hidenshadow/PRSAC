$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$MediumRoot = Join-Path $Root "runs\sac_modified\ldac_v2_medium_seed0_1_20260621"
$HardScript = Join-Path $ScriptDir "run_sac_ldac_v2_hard_seed0_1.ps1"
$WatcherLog = Join-Path $MediumRoot "watcher_hard_decision.log"

$Levels = @("level1_medium", "level2_medium", "level3_medium")
$Seeds = @(0, 1)

New-Item -ItemType Directory -Force -Path $MediumRoot | Out-Null
"Hard watcher started: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append

while ($true) {
    $missing = @()
    foreach ($level in $Levels) {
        foreach ($seed in $Seeds) {
            $summary = Join-Path $MediumRoot (Join-Path $level (Join-Path ("seed" + $seed) "shock_recovery_summary.csv"))
            if (-not (Test-Path $summary)) {
                $missing += "$level/seed$seed"
            }
        }
    }

    if ($missing.Count -eq 0) {
        "Medium experiments complete: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
        break
    }

    "Waiting for medium outputs: missing=$($missing -join ', ') time=$(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
    Start-Sleep -Seconds 300
}

"Starting hard experiments: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $HardScript
) -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $MediumRoot "hard_scheduler_stdout.log") `
    -RedirectStandardError (Join-Path $MediumRoot "hard_scheduler_stderr.log") | Out-Null
