param(
    [string]$CaseDir = "",
    [string]$IsaacLabRoot = "<DESKTOP>\ISAAC\IsaacLab",
    [string]$IsaacPython = "<DESKTOP>\ISAAC\.venv\Scripts\python.exe",
    [double]$MapSizeM = 0.0,
    [double]$MapResolution = 0.0,
    [int]$EnableSimpleAvoid = 1,
    [double]$WaypointSpacing = 0.5,
    [double]$MaxSpeed = 0.8,
    [double]$MaxSteer = 0.35,
    [int]$MaxSteps = 0,
    [string]$Device = "",
    [ValidateSet("none", "app", "imports", "scene", "post-scene")]
    [string]$SmokeStage = "none",
    [ValidateSet("full", "terrain", "robot")]
    [string]$SceneContent = "full",
    [ValidateSet("physical", "visual-only", "usd")]
    [string]$RobotUrdfMode = "usd",
    [string]$RobotUsdFile = "",
    [ValidateSet("controller", "kinematic")]
    [string]$ReplayMode = "kinematic",
    [double]$KinematicSpeedMps = 0.8,
    [int]$FastExit = 1,
    [string]$SaveUsdFile = "",
    [int]$ExportUsdOnly = 0,
    [switch]$Headless,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Format-ArgNumber([double]$Value) {
    return $Value.ToString("G17", [Globalization.CultureInfo]::InvariantCulture)
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($CaseDir)) {
    $CaseDir = Join-Path $ProjectRoot "exports\isaac_policy_case_study\level2_medium_valt_sac_seed0_task0"
}

$CaseDir = (Resolve-Path $CaseDir).Path
$IsaacLabRoot = (Resolve-Path $IsaacLabRoot).Path

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    throw "Isaac Python not found: $IsaacPython"
}

$IsaacScript = Join-Path $IsaacLabRoot "scripts\custom\create_moon_planning_robot_waypoint.py"
if (-not (Test-Path -LiteralPath $IsaacScript)) {
    throw "Isaac waypoint script not found: $IsaacScript"
}

$TerrainFile = Join-Path $CaseDir "terrain_heightfield.npy"
$WaypointFile = Join-Path $CaseDir "policy_waypoints.csv"
$MetadataFile = Join-Path $CaseDir "metadata.json"

foreach ($path in @($TerrainFile, $WaypointFile, $MetadataFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required case-study file not found: $path"
    }
}

$Metadata = Get-Content -LiteralPath $MetadataFile -Raw | ConvertFrom-Json
if ($MapSizeM -le 0.0) {
    $MapSizeM = [double]$Metadata.map_size_m
}
if ($MapSizeM -le 0.0) {
    $MapSizeM = 20.0
}

if ($MapResolution -le 0.0) {
    $rows = 0
    if ($Metadata.grid_shape -and $Metadata.grid_shape.Count -gt 0) {
        $rows = [int]$Metadata.grid_shape[0]
    }
    if ($rows -gt 0) {
        $MapResolution = $MapSizeM / [double]$rows
    } else {
        $MapResolution = 0.25
    }
}

if (-not [string]::IsNullOrWhiteSpace($SaveUsdFile)) {
    if ([System.IO.Path]::IsPathRooted($SaveUsdFile)) {
        $SaveUsdFile = [System.IO.Path]::GetFullPath($SaveUsdFile)
    } else {
        $SaveUsdFile = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SaveUsdFile))
    }
}

$LogDir = Join-Path $CaseDir "isaac_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("robot_waypoint_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$IsaacArgs = @(
    $IsaacScript,
    "--terrain-file", $TerrainFile,
    "--waypoint-file", $WaypointFile,
    "--waypoint-format", "global",
    "--map-size-m", (Format-ArgNumber $MapSizeM),
    "--map-resolution", (Format-ArgNumber $MapResolution),
    "--terrain-dither-scale", "0.0",
    "--spawn-at-first-waypoint", "1",
    "--spawn-z", "0.45",
    "--enable-simple-avoid", "$EnableSimpleAvoid",
    "--waypoint-spacing", (Format-ArgNumber $WaypointSpacing),
    "--max-speed", (Format-ArgNumber $MaxSpeed),
    "--max-steer", (Format-ArgNumber $MaxSteer),
    "--max-steps", "$MaxSteps",
    "--fast-exit", "$FastExit",
    "--export-usd-only", "$ExportUsdOnly",
    "--smoke-stage", "$SmokeStage",
    "--scene-content", "$SceneContent",
    "--robot-urdf-mode", "$RobotUrdfMode",
    "--replay-mode", "$ReplayMode",
    "--kinematic-speed-mps", (Format-ArgNumber $KinematicSpeedMps)
)

if (-not [string]::IsNullOrWhiteSpace($RobotUsdFile)) {
    $IsaacArgs += @("--robot-usd-file", $RobotUsdFile)
}

if (-not [string]::IsNullOrWhiteSpace($SaveUsdFile)) {
    $IsaacArgs += @("--save-usd-file", $SaveUsdFile)
}

if ($Headless) {
    $IsaacArgs += "--headless"
}

if (-not [string]::IsNullOrWhiteSpace($Device)) {
    $IsaacArgs += @("--device", $Device)
}

if ($ExtraArgs) {
    $IsaacArgs += $ExtraArgs
}

Write-Host "[rl_weight_planner] case: $CaseDir"
Write-Host "[rl_weight_planner] terrain: $TerrainFile"
Write-Host "[rl_weight_planner] waypoints: $WaypointFile"
Write-Host "[rl_weight_planner] map_size_m=$MapSizeM map_resolution=$MapResolution"
Write-Host "[rl_weight_planner] log: $LogFile"
Write-Host "[rl_weight_planner] isaac python: $IsaacPython"

if ($DryRun) {
    Write-Host "[rl_weight_planner] dry run only. Command:"
    Write-Host "& `"$IsaacPython`" $($IsaacArgs -join ' ')"
    exit 0
}

Push-Location $IsaacLabRoot
try {
    & $IsaacPython @IsaacArgs 2>&1 | Tee-Object -FilePath $LogFile
} finally {
    Pop-Location
}

