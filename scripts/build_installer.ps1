[CmdletBinding()]
param(
    [string]$IsccExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$standaloneRoot = Join-Path $projectRoot "build\windows\AI-Biaozhu.dist"
$standaloneExe = Join-Path $standaloneRoot "AI-Biaozhu.exe"
$workerExe = Join-Path $standaloneRoot "AI-Biaozhu-Worker.exe"
$legacyRoot = Join-Path $standaloneRoot "third_party\yolov5"
$tagMarker = Join-Path $legacyRoot ".ai-biaozhu-yolov5-tag"
$runtimeLicenses = Join-Path $standaloneRoot "THIRD_PARTY_LICENSES"
$issPath = Join-Path $projectRoot "packaging\ai_biaozhu.iss"
$modelSeed = Join-Path $standaloneRoot "model-seed\yolo26s.pt"
$modelSeedSha256 = "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B"
$modelSeedSize = 20422725

foreach ($required in @(
    $standaloneExe,
    $workerExe,
    $tagMarker,
    $modelSeed,
    $issPath,
    (Join-Path $standaloneRoot "LICENSE"),
    (Join-Path $standaloneRoot "THIRD_PARTY_NOTICES.md"),
    (Join-Path $runtimeLicenses "index.json"),
    (Join-Path $runtimeLicenses "CPython-3.11-PSF-LICENSE.txt"),
    (Join-Path $runtimeLicenses "YOLOv5-v7.0-GPL-3.0.txt")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required installer input is missing: $required"
    }
}
if (Get-ChildItem -LiteralPath $legacyRoot -Directory -Force -Recurse |
    Where-Object { $_.Name -eq ".git" }) {
    throw "Standalone YOLOv5 runtime contains a forbidden .git directory."
}
if (Get-ChildItem -LiteralPath $standaloneRoot -File -Force -Recurse |
    Where-Object {
        $_.Extension -in @(".pt", ".onnx", ".engine") -and
        $_.FullName -ne $modelSeed
    }) {
    throw "Standalone unexpectedly contains model weights or exported model artifacts."
}
$seedItem = Get-Item -LiteralPath $modelSeed
$seedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelSeed).Hash
if ($seedItem.Length -ne $modelSeedSize -or $seedHash -ne $modelSeedSha256) {
    throw "Standalone yolo26s.pt seed does not match its locked size/SHA-256."
}

$lock = Get-Content -LiteralPath `
    (Join-Path $projectRoot "third_party\yolov5.lock.json") `
    -Raw -Encoding UTF8 | ConvertFrom-Json
$tagLines = @(Get-Content -LiteralPath $tagMarker -Encoding UTF8)
$actualTag = $tagLines[0].Trim()
if ($actualTag -ne [string]$lock.ref) {
    throw "Bundled YOLOv5 tag is $actualTag; expected $($lock.ref)."
}
$actualCommit = (
    $tagLines |
        Where-Object { $_.StartsWith("commit=") } |
        Select-Object -First 1
) -replace "^commit=", ""
if ($actualCommit -ne [string]$lock.commit) {
    throw "Bundled YOLOv5 commit marker does not match $($lock.commit)."
}

if (-not $IsccExe) {
    $isccCandidates = New-Object System.Collections.Generic.List[string]
    if ($env:AI_BIAOZHU_ISCC) {
        $isccCandidates.Add($env:AI_BIAOZHU_ISCC)
    }
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) {
        $isccCandidates.Add($command.Source)
    }
    foreach ($base in @(
        ${env:ProgramFiles(x86)},
        $env:ProgramFiles,
        $env:LOCALAPPDATA
    )) {
        if ($base) {
            $isccCandidates.Add((Join-Path $base "Inno Setup 6\ISCC.exe"))
            $isccCandidates.Add((Join-Path $base "Programs\Inno Setup 6\ISCC.exe"))
        }
    }
    foreach ($keyPath in @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
    )) {
        $item = Get-ItemProperty -LiteralPath $keyPath -ErrorAction SilentlyContinue
        if ($item -and $item.InstallLocation) {
            $isccCandidates.Add((Join-Path $item.InstallLocation "ISCC.exe"))
        }
    }
    $IsccExe = $isccCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $IsccExe) {
        throw "Inno Setup 6 ISCC.exe was not found. Pass -IsccExe explicitly."
    }
}
if (-not (Test-Path -LiteralPath $IsccExe -PathType Leaf)) {
    throw "ISCC.exe does not exist: $IsccExe"
}
$isccVersion = (Get-Item -LiteralPath $IsccExe).VersionInfo.ProductVersion
$versionMatch = [regex]::Match([string]$isccVersion, "\d+\.\d+(?:\.\d+){0,2}")
if (
    -not $versionMatch.Success -or
    ([version]$versionMatch.Value) -eq [version]"0.0.0.0"
) {
    # Official Inno Setup 6.7.3 deliberately reports 0.0.0.0 on ISCC.exe.
    # Its installed uninstaller retains the actual product version.
    $uninstaller = Join-Path (Split-Path -Parent $IsccExe) "unins000.exe"
    if (Test-Path -LiteralPath $uninstaller -PathType Leaf) {
        $isccVersion = (Get-Item -LiteralPath $uninstaller).VersionInfo.ProductVersion
        $versionMatch = [regex]::Match(
            [string]$isccVersion,
            "\d+\.\d+(?:\.\d+){0,2}"
        )
    }
}
if ($versionMatch.Success) {
    if ([version]$versionMatch.Value -lt [version]"6.3.0") {
        throw "Inno Setup 6.3 or newer is required; found version $isccVersion"
    }
} else {
    # The actual .iss requires the 6.3+ ``x64compatible`` architecture token.
    # A compiler with no trustworthy version metadata is accepted here only
    # provisionally; successful compilation below is the capability check.
    Write-Warning (
        "Could not read a trustworthy Inno Setup version. " +
        "The compiler must successfully compile the 6.3+ x64compatible script."
    )
}

$pyproject = Get-Content -LiteralPath (Join-Path $projectRoot "pyproject.toml") `
    -Encoding UTF8
$versionLine = $pyproject |
    Where-Object { $_ -match '^version\s*=\s*"([^"]+)"\s*$' } |
    Select-Object -First 1
if (-not $versionLine) {
    throw "Could not read the application version from pyproject.toml"
}
$appVersion = [regex]::Match(
    $versionLine,
    '^version\s*=\s*"([^"]+)"\s*$'
).Groups[1].Value

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "dist") |
    Out-Null
& $IsccExe "/Qp" "/DMyAppVersion=$appVersion" $issPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed."
}

$installer = Join-Path $projectRoot `
    "dist\AI-Biaozhu-Maintenance-Setup-$appVersion-x64.exe"
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "ISCC completed but the expected installer is missing: $installer"
}
Write-Host "Installer build completed: $installer"
