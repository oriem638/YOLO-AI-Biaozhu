[CmdletBinding()]
param(
    [string]$CondaExe = "",
    [switch]$UpdateExisting,
    [switch]$RequireGpu
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
    $condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaCommand) {
        $CondaExe = $condaCommand.Source
    } elseif (Test-Path -LiteralPath "E:\tools\anacondass\Scripts\conda.exe") {
        $CondaExe = "E:\tools\anacondass\Scripts\conda.exe"
    } else {
        throw "conda.exe was not found. Pass its full path with -CondaExe."
    }
}
if (-not (Test-Path -LiteralPath $CondaExe -PathType Leaf)) {
    throw "conda.exe does not exist: $CondaExe"
}

$environmentJson = (& $CondaExe env list --json | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Could not list Conda environments."
}
$environmentNames = ($environmentJson | ConvertFrom-Json).envs |
    ForEach-Object { Split-Path -Leaf $_ }

Push-Location $projectRoot
try {
    if ($environmentNames -contains "yolo") {
        if (-not $UpdateExisting) {
            Write-Host (
                "Conda environment 'yolo' already exists. " +
                "Use -UpdateExisting to reconcile it with environment.yml."
            )
        } else {
            & $CondaExe env update --name yolo --file environment.yml --prune
            if ($LASTEXITCODE -ne 0) {
                throw "Updating the yolo environment failed."
            }
        }
    } else {
        & $CondaExe env create --file environment.yml
        if ($LASTEXITCODE -ne 0) {
            throw "Creating the yolo environment failed."
        }
    }

    & $CondaExe run --no-capture-output --name yolo python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "The yolo environment failed pip check."
    }
    $probeArguments = @(
        "run",
        "--no-capture-output",
        "--name",
        "yolo",
        "python",
        (Join-Path $PSScriptRoot "probe_yolo_environment.py")
    )
    if ($RequireGpu) {
        $probeArguments += "--require-gpu"
    }
    & $CondaExe @probeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The yolo environment version/GPU probe failed."
    }

    & (Join-Path $PSScriptRoot "fetch_yolov5.ps1")
} finally {
    Pop-Location
}

Write-Host "The locked yolo development environment is ready."
