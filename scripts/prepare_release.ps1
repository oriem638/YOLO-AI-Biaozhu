[CmdletBinding()]
param(
    [string]$CondaExe = "",
    [string]$OutputDirectory = "",
    [string]$SourceMirror = "",
    [switch]$ReplaceSourceMirror,
    [switch]$SourceOnly,
    [switch]$SkipStandaloneZip,
    [switch]$CleanOutput,
    [switch]$AllowPartialInstallerValidation
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputDirectory) {
    $projectLeaf = Split-Path -Leaf $projectRoot
    $OutputDirectory = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot "..\$projectLeaf-outputs")
    )
}
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
if (-not (Test-Path -LiteralPath $CondaExe -PathType Leaf)) {
    throw "conda.exe does not exist: $CondaExe"
}

$arguments = @(
    "run",
    "--no-capture-output",
    "--name",
    "yolo",
    "python",
    (Join-Path $PSScriptRoot "prepare_release.py"),
    "--project-root",
    $projectRoot,
    "--output",
    ([System.IO.Path]::GetFullPath($OutputDirectory))
)
if ($SourceMirror) {
    $arguments += @(
        "--source-mirror",
        ([System.IO.Path]::GetFullPath($SourceMirror))
    )
}
if ($ReplaceSourceMirror) {
    $arguments += "--replace-source-mirror"
}
if ($SourceOnly) {
    $arguments += "--source-only"
}
if ($SkipStandaloneZip) {
    $arguments += "--skip-standalone-zip"
}
if ($CleanOutput) {
    $arguments += "--clean-output"
}
if ($AllowPartialInstallerValidation) {
    $arguments += "--allow-partial-installer-validation"
}

& $CondaExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Preparing audited release deliverables failed."
}
