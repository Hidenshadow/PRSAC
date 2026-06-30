param(
    [string]$CurrentRoot = "runs\bvr\advantage_gate_easy_2seed_20260620",
    [string]$NextRoot = "runs\bvr\belief_safe_easy_2seed_20260620",
    [int]$ExpectedDone = 6,
    [int]$PollSeconds = 120,
    [int]$MaxParallel = 6,
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$LogPath = Join-Path "logs" ("bvr_belief_safe_rerun_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Get-DoneCount {
    if (!(Test-Path $CurrentRoot)) {
        return 0
    }
    return (Get-ChildItem $CurrentRoot -Recurse -Filter "BVR_DONE.txt" -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Get-FailedCount {
    if (!(Test-Path $CurrentRoot)) {
        return 0
    }
    return (Get-ChildItem $CurrentRoot -Recurse -Filter "BVR_FAILED.txt" -ErrorAction SilentlyContinue | Measure-Object).Count
}

Write-Log "waiting for current BVR pilot: $CurrentRoot"
while ($true) {
    $done = Get-DoneCount
    $failed = Get-FailedCount
    Write-Log ("current progress done={0}/{1} failed={2}" -f $done, $ExpectedDone, $failed)
    if ($done -ge $ExpectedDone) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "starting belief-safe BVR rerun -> $NextRoot"
& $Python scripts\run_bvr_matrix_incremental.py `
    --output-root $NextRoot `
    --levels level1 level2 level3 `
    --difficulties easy `
    --seeds 0 1 `
    --max-parallel $MaxParallel `
    --num-iterations 20 `
    --eval-episodes 300 `
    --rollout-episodes-per-iteration 64 `
    --max-candidate-sets-per-iteration 128 `
    --selection-margin 0.0005 `
    --belief-safety-penalty 1.0 `
    --belief-cost-margin 0.02 `
    --belief-constraint-margin 0.02 2>&1 | ForEach-Object { Write-Log ("  " + $_) }

if ($LASTEXITCODE -ne 0) {
    Write-Log ("belief-safe BVR rerun failed with exit code {0}" -f $LASTEXITCODE)
    exit $LASTEXITCODE
}
Write-Log "belief-safe BVR rerun finished"
