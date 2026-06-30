param(
    [string]$NonLearningRunDir = "runs\nonlearning_planner_baselines\5seeds_incremental_full_20260619_043515",
    [int]$ExpectedChunks = 45,
    [int]$PollSeconds = 120,
    [int]$NonLearningProcessId = 0,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputRoot = "runs\bvr\full_5seed_after_nonlearning_20260619",
    [int]$MaxParallel = 0,
    [int]$NumIterations = 20,
    [int]$EvalEpisodes = 300,
    [int]$RolloutEpisodes = 64,
    [int]$MaxCandidateSets = 128,
    [int]$VerifierEpochs = 5,
    [int]$VerifierBatchSize = 32,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($MaxParallel -le 0) {
    $logicalCpu = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    $MaxParallel = [Math]::Max(1, [Math]::Floor($logicalCpu * 0.8))
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PYTORCH_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$LogPath = Join-Path "logs" ("post_nonlearning_bvr_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "s"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Get-DoneCount {
    $chunks = Join-Path $NonLearningRunDir "chunks"
    if (!(Test-Path $chunks)) {
        return 0
    }
    return (Get-ChildItem $chunks -Filter "*.done" -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Test-NonLearningProcessRunning {
    if ($NonLearningProcessId -le 0) {
        return $false
    }
    $proc = Get-Process -Id $NonLearningProcessId -ErrorAction SilentlyContinue
    return ($null -ne $proc)
}

function Test-FinalLogLine {
    $stdout = Join-Path $NonLearningRunDir "evaluation_stdout.log"
    if (!(Test-Path $stdout)) {
        return $false
    }
    $match = Select-String -Path $stdout -Pattern "Incremental non-learning baselines finished" -Quiet
    return [bool]$match
}

function Invoke-LoggedCommand {
    param([string]$FilePath, [string[]]$Arguments)
    Write-Log ("running: {0} {1}" -f $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Log ("  " + $_) }
    if ($LASTEXITCODE -ne 0) {
        throw ("command failed with exit code {0}: {1}" -f $LASTEXITCODE, $FilePath)
    }
}

function Upload-NonLearningResults {
    Write-Log "uploading non-learning baseline results from $NonLearningRunDir"
    Invoke-LoggedCommand "git" @("add", "-f", "--", $NonLearningRunDir)

    $prefix = $NonLearningRunDir.Replace("\", "/").TrimEnd("/") + "/"
    $cached = @(git diff --cached --name-only)
    $outside = @($cached | Where-Object { ($_ -notlike "$prefix*") -and ($_ -ne $prefix.TrimEnd("/")) })
    if ($outside.Count -gt 0) {
        Write-Log "refusing to commit because staged files outside non-learning result dir were found:"
        foreach ($path in $outside) {
            Write-Log ("  " + $path)
        }
        throw "unexpected staged files outside non-learning results"
    }

    & git diff --cached --quiet -- $NonLearningRunDir
    if ($LASTEXITCODE -eq 0) {
        Write-Log "no non-learning result changes to commit"
        return
    }

    Invoke-LoggedCommand "git" @("commit", "-m", "Upload non-learning planner baseline results")

    $pushed = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Write-Log "git push attempt $attempt"
            Invoke-LoggedCommand "git" @("push", "origin", "main")
            $pushed = $true
            break
        } catch {
            Write-Log ("git push attempt {0} failed: {1}" -f $attempt, $_.Exception.Message)
            if ($attempt -lt 3) {
                Start-Sleep -Seconds 60
            }
        }
    }
    if (!$pushed) {
        Write-Log "non-learning results were committed locally but push failed after retries"
    }
}

function New-BvrTasks {
    $tasks = @()
    $levels = @("level1", "level2", "level3")
    $difficulties = @("easy", "medium", "hard")
    foreach ($level in $levels) {
        foreach ($difficulty in $difficulties) {
            $scenario = "{0}_{1}" -f $level, $difficulty
            foreach ($seed in 0..4) {
                $source = Join-Path "runs\rl_baselines\ppo" ("{0}_shock_recovery_5seeds\seed{1}" -f $scenario, $seed)
                $output = Join-Path $OutputRoot (Join-Path $scenario ("seed{0}" -f $seed))
                $tasks += [PSCustomObject]@{
                    Scenario = $scenario
                    Seed = $seed
                    Source = $source
                    Output = $output
                }
            }
        }
    }
    return $tasks
}

function Start-BvrTask {
    param([object]$Task)
    New-Item -ItemType Directory -Force -Path $Task.Output | Out-Null
    $stdout = Join-Path $Task.Output "stdout.log"
    $stderr = Join-Path $Task.Output "stderr.log"
    $args = @(
        "scripts\train_bvr.py",
        "--source-run-dir", $Task.Source,
        "--output-dir", $Task.Output,
        "--device", "cpu",
        "--num-iterations", "$NumIterations",
        "--eval-episodes", "$EvalEpisodes",
        "--rollout-episodes-per-iteration", "$RolloutEpisodes",
        "--max-candidate-sets-per-iteration", "$MaxCandidateSets",
        "--verifier-epochs", "$VerifierEpochs",
        "--verifier-batch-size", "$VerifierBatchSize"
    )
    Write-Log ("starting BVR {0} seed{1}" -f $Task.Scenario, $Task.Seed)
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    return [PSCustomObject]@{
        Process = $proc
        Task = $Task
        Stdout = $stdout
        Stderr = $stderr
        StartTime = Get-Date
    }
}

function Run-BvrMatrix {
    $tasks = @(New-BvrTasks)
    Write-Log ("BVR task count={0}, max_parallel={1}, output_root={2}" -f $tasks.Count, $MaxParallel, $OutputRoot)
    $running = @()
    $next = 0
    $completed = 0
    $failed = 0
    $skipped = 0

    while (($next -lt $tasks.Count) -or ($running.Count -gt 0)) {
        while (($running.Count -lt $MaxParallel) -and ($next -lt $tasks.Count)) {
            $task = $tasks[$next]
            $next += 1
            $doneMarker = Join-Path $task.Output "BVR_DONE.txt"
            if (Test-Path $doneMarker) {
                $skipped += 1
                Write-Log ("skipping existing BVR_DONE {0} seed{1}" -f $task.Scenario, $task.Seed)
                continue
            }
            if (!(Test-Path $task.Source)) {
                $failed += 1
                New-Item -ItemType Directory -Force -Path $task.Output | Out-Null
                "missing source run: $($task.Source)" | Set-Content -Path (Join-Path $task.Output "BVR_FAILED.txt") -Encoding UTF8
                Write-Log ("missing source for BVR {0} seed{1}: {2}" -f $task.Scenario, $task.Seed, $task.Source)
                continue
            }
            $running += Start-BvrTask -Task $task
        }

        Start-Sleep -Seconds 30
        $stillRunning = @()
        foreach ($entry in $running) {
            $entry.Process.Refresh()
            if ($entry.Process.HasExited) {
                $task = $entry.Task
                $exitCode = $entry.Process.ExitCode
                if ($exitCode -eq 0) {
                    $completed += 1
                    ("completed {0}" -f (Get-Date -Format "s")) | Set-Content -Path (Join-Path $task.Output "BVR_DONE.txt") -Encoding UTF8
                    Write-Log ("completed BVR {0} seed{1}" -f $task.Scenario, $task.Seed)
                } else {
                    $failed += 1
                    ("failed exit_code={0} time={1}" -f $exitCode, (Get-Date -Format "s")) | Set-Content -Path (Join-Path $task.Output "BVR_FAILED.txt") -Encoding UTF8
                    Write-Log ("failed BVR {0} seed{1} exit_code={2}" -f $task.Scenario, $task.Seed, $exitCode)
                }
            } else {
                $stillRunning += $entry
            }
        }
        $running = $stillRunning
        Write-Log ("BVR progress completed={0} failed={1} skipped={2} running={3} queued={4}" -f $completed, $failed, $skipped, $running.Count, ($tasks.Count - $next))
    }

    Write-Log ("BVR matrix finished completed={0} failed={1} skipped={2}" -f $completed, $failed, $skipped)
}

Write-Log "post-nonlearning watcher started"
Write-Log ("nonlearning_dir={0}, expected_chunks={1}, pid={2}" -f $NonLearningRunDir, $ExpectedChunks, $NonLearningProcessId)
Write-Log ("python={0}, max_parallel={1}" -f $Python, $MaxParallel)

if ($DryRun) {
    $tasks = @(New-BvrTasks)
    Write-Log ("dry run: task_count={0}, first_task={1} seed{2}" -f $tasks.Count, $tasks[0].Scenario, $tasks[0].Seed)
    exit 0
}

while ($true) {
    $done = Get-DoneCount
    $running = Test-NonLearningProcessRunning
    $finalLine = Test-FinalLogLine
    Write-Log ("waiting non-learning done={0}/{1} running={2} final_log={3}" -f $done, $ExpectedChunks, $running, $finalLine)
    if (($done -ge $ExpectedChunks) -and (!$running) -and $finalLine) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Upload-NonLearningResults
Run-BvrMatrix
Write-Log "post-nonlearning watcher finished"
