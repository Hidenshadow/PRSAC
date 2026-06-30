$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$EasyRoot = Join-Path $Root "runs\bvr\same_protocol_easy_2seed_gate_20260620"
$DecisionDir = Join-Path $EasyRoot "decision"
$MediumScript = Join-Path $ScriptDir "run_bvr_same_protocol_medium_2seed.ps1"
$WatcherLog = Join-Path $EasyRoot "watcher_decision.log"

$Levels = @("level1_easy", "level2_easy", "level3_easy")
$Seeds = @(0, 1)

New-Item -ItemType Directory -Force -Path $DecisionDir | Out-Null
"Watcher started: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append

while ($true) {
    $missing = @()
    foreach ($level in $Levels) {
        foreach ($seed in $Seeds) {
            $summary = Join-Path $EasyRoot (Join-Path $level (Join-Path ("seed" + $seed) "shock_recovery_summary.csv"))
            if (-not (Test-Path $summary)) {
                $missing += "$level/seed$seed"
            }
        }
    }

    if ($missing.Count -eq 0) {
        "Easy experiments complete: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
        break
    }

    "Waiting for easy outputs: missing=$($missing -join ', ') time=$(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
    Start-Sleep -Seconds 300
}

$summaryArgs = @(
    (Join-Path $ScriptDir "summarize_bvr_same_protocol.py"),
    "--bvr-root", $EasyRoot,
    "--levels", "level1_easy", "level2_easy", "level3_easy",
    "--seeds", "0", "1",
    "--output-dir", $DecisionDir
)
& $Python @summaryArgs *> (Join-Path $DecisionDir "summarize_stdout_stderr.log")
$exitCode = $LASTEXITCODE
"Decision summarizer exit code: $exitCode" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append

if ($exitCode -eq 0) {
    "Positive easy result. Starting medium experiments: $(Get-Date -Format s)" | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $MediumScript
    ) -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $EasyRoot "medium_scheduler_stdout.log") `
        -RedirectStandardError (Join-Path $EasyRoot "medium_scheduler_stderr.log") | Out-Null
} else {
    "Easy result not positive enough. Medium experiments will not start. Switch direction." | Out-File -FilePath $WatcherLog -Encoding utf8 -Append
}
