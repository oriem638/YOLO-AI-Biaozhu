[CmdletBinding()]
param(
    [string]$CondaExe = "",
    [switch]$IncludeGpu,
    [switch]$IncludeDocker
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$yoloConfig = Join-Path $projectRoot "build\yolo-config"
New-Item -ItemType Directory -Force -Path $yoloConfig | Out-Null
$env:YOLO_CONFIG_DIR = $yoloConfig
if (-not $CondaExe) {
    $candidate = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($candidate) {
        $CondaExe = $candidate.Source
    } elseif (Test-Path -LiteralPath "E:\tools\anacondass\Scripts\conda.exe") {
        $CondaExe = "E:\tools\anacondass\Scripts\conda.exe"
    } else {
        throw "conda.exe was not found. Pass its full path with -CondaExe."
    }
}

Push-Location $projectRoot
try {
    & (Join-Path $PSScriptRoot "validate_packaging.ps1")

    & $CondaExe run --no-capture-output --name yolo python -m ruff check src tests
    if ($LASTEXITCODE -ne 0) {
        throw "Ruff checks failed."
    }

    $pytestArguments = @("-m", "pytest", "-q")
    if (-not $IncludeGpu -and -not $IncludeDocker) {
        $pytestArguments += @("-m", "not gpu and not docker and not device")
    } elseif (-not $IncludeGpu) {
        $pytestArguments += @("-m", "not gpu and not device")
    } elseif (-not $IncludeDocker) {
        $pytestArguments += @("-m", "not docker and not device")
    } else {
        $pytestArguments += @("-m", "not device")
    }
    & $CondaExe run --no-capture-output --name yolo python @pytestArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pytest failed."
    }
} finally {
    Pop-Location
}
