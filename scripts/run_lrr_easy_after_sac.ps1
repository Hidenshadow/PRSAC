param(
    [string]$WatchRoot = "runs\rl_baselines\sac\level3_hard_shock_recovery_5seeds",
    [int[]]$RequiredSeeds = @(0, 1, 2, 3),
    [int]$TargetRecoveryStep = 20480,
    [int]$PollSeconds = 120,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputRoot = "runs\lrr\easy_1seed_after_sac",
    [int]$LrrIterations = 20,
    [int]$LrrEvalEpisodes = 300,
    [int]$LrrRolloutEpisodes = 20,
    [int]$LrrRepairEpochs = 5,
    [int]$LrrBatchSize = 64,
    [double]$DeadlineHours = 0,
    [switch]$IncludePriorityFollowups,
    [switch]$SkipIfDone
)

$ErrorActionPreference = "Stop"
$Deadline = $null
if ($DeadlineHours -gt 0) {
    $Deadline = (Get-Date).AddHours($DeadlineHours)
}

function Get-MaxRecoveryStep {
    param([string]$SeedDir)
    $checkpointDir = Join-Path $SeedDir "checkpoints"
    if (!(Test-Path $checkpointDir)) {
        return 0
    }
    $maxStep = 0
    Get-ChildItem $checkpointDir -Filter "checkpoint_recovery_step_*.pt" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match '_(\d+)\.pt$') {
            $step = [int]$Matches[1]
            if ($step -gt $maxStep) {
                $maxStep = $step
            }
        }
    }
    return $maxStep
}

function Test-RequiredSeedsDone {
    $status = @()
    foreach ($seed in $RequiredSeeds) {
        $seedDir = Join-Path $WatchRoot ("seed{0}" -f $seed)
        $maxStep = Get-MaxRecoveryStep -SeedDir $seedDir
        $status += [PSCustomObject]@{
            Seed = $seed
            MaxRecoveryStep = $maxStep
            Done = ($maxStep -ge $TargetRecoveryStep)
        }
    }
    return $status
}

function Invoke-LrrRun {
    param(
        [string]$Scenario,
        [string]$SourceRunDir,
        [string]$OutputDir
    )
    $doneMarker = Join-Path $OutputDir "LRR_DONE.txt"
    if ($SkipIfDone -and (Test-Path $doneMarker)) {
        Write-Output "$(Get-Date -Format s): skipping $Scenario because $doneMarker exists"
        return
    }
    if ($null -ne $Deadline -and (Get-Date) -ge $Deadline) {
        Write-Output "$(Get-Date -Format s): skipping $Scenario because deadline $($Deadline.ToString('s')) has passed"
        return
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Output "$(Get-Date -Format s): starting LRR $Scenario -> $OutputDir"
    & $Python scripts\train_lrr.py `
        --source-run-dir $SourceRunDir `
        --output-dir $OutputDir `
        --num-iterations $LrrIterations `
        --eval-episodes $LrrEvalEpisodes `
        --rollout-episodes-per-iteration $LrrRolloutEpisodes `
        --max-repair-states-per-iteration $LrrRolloutEpisodes `
        --repair-epochs $LrrRepairEpochs `
        --repair-batch-size $LrrBatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "LRR $Scenario failed with exit code $LASTEXITCODE"
    }
    "completed $(Get-Date -Format s)" | Set-Content -Path $doneMarker -Encoding UTF8
    Write-Output "$(Get-Date -Format s): completed LRR $Scenario"
}

Write-Output "$(Get-Date -Format s): waiting for SAC seeds $($RequiredSeeds -join ',') to reach recovery step $TargetRecoveryStep"
if ($null -ne $Deadline) {
    Write-Output "$(Get-Date -Format s): deadline before starting new LRR runs is $($Deadline.ToString('s'))"
}
while ($true) {
    $status = Test-RequiredSeedsDone
    $statusLine = ($status | ForEach-Object { "seed$($_.Seed)=$($_.MaxRecoveryStep)" }) -join " "
    Write-Output "$(Get-Date -Format s): $statusLine"
    if (($status | Where-Object { -not $_.Done }).Count -eq 0) {
        break
    }
    if ($null -ne $Deadline -and (Get-Date) -ge $Deadline) {
        Write-Output "$(Get-Date -Format s): deadline passed while waiting for SAC; exiting without starting LRR"
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Output "$(Get-Date -Format s): required SAC seeds complete; starting LRR easy single-seed validation"

$runs = @(
    @{
        Scenario = "level1_easy"
        Source = "runs\rl_baselines\ppo\level1_easy_shock_recovery_5seeds\seed0"
        Output = Join-Path $OutputRoot "level1_easy\seed0"
    },
    @{
        Scenario = "level2_easy"
        Source = "runs\rl_baselines\ppo\level2_easy_shock_recovery_5seeds\seed0"
        Output = Join-Path $OutputRoot "level2_easy\seed0"
    },
    @{
        Scenario = "level3_easy"
        Source = "runs\rl_baselines\ppo\level3_easy_shock_recovery_5seeds\seed0"
        Output = Join-Path $OutputRoot "level3_easy\seed0"
    }
)

if ($IncludePriorityFollowups) {
    $runs += @(
        @{
            Scenario = "level1_hard"
            Source = "runs\rl_baselines\ppo\level1_hard_shock_recovery_5seeds\seed0"
            Output = Join-Path $OutputRoot "level1_hard\seed0"
        },
        @{
            Scenario = "level2_medium"
            Source = "runs\rl_baselines\ppo\level2_medium_shock_recovery_5seeds\seed0"
            Output = Join-Path $OutputRoot "level2_medium\seed0"
        },
        @{
            Scenario = "level3_medium"
            Source = "runs\rl_baselines\ppo\level3_medium_shock_recovery_5seeds\seed0"
            Output = Join-Path $OutputRoot "level3_medium\seed0"
        }
    )
}

foreach ($run in $runs) {
    Invoke-LrrRun -Scenario $run.Scenario -SourceRunDir $run.Source -OutputDir $run.Output
}

Write-Output "$(Get-Date -Format s): all LRR easy single-seed validation runs complete"
