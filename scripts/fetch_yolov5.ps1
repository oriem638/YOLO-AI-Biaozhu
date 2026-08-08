[CmdletBinding()]
param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$lockPath = Join-Path $projectRoot "third_party\yolov5.lock.json"
$tagFileName = ".ai-biaozhu-yolov5-tag"

if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    throw "Missing YOLOv5 lock file: $lockPath"
}
$lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
$expectedTag = [string]$lock.ref
$expectedCommit = [string]$lock.commit
$repository = [string]$lock.repository
if (-not $expectedTag -or $expectedCommit -notmatch "^[0-9a-f]{40}$" -or -not $repository) {
    throw "Invalid YOLOv5 lock file: $lockPath"
}

if (-not $Destination) {
    $Destination = Join-Path $projectRoot "third_party\runtime\yolov5"
}
$destinationFull = [System.IO.Path]::GetFullPath($Destination)
$projectFull = [System.IO.Path]::GetFullPath($projectRoot)
$projectPrefix = $projectFull.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
if (-not $destinationFull.StartsWith(
    $projectPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Destination must be inside the project directory: $destinationFull"
}

function Write-TagLock {
    param([Parameter(Mandatory = $true)][string]$Root)

    $marker = Join-Path $Root $tagFileName
    $contents = "$expectedTag`ncommit=$expectedCommit`nrepository=$repository`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($marker, $contents, $utf8NoBom)
}

function Assert-RequiredFiles {
    param([Parameter(Mandatory = $true)][string]$Root)

    foreach ($name in @("train.py", "detect.py", "export.py", "LICENSE")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $name) -PathType Leaf)) {
            throw "YOLOv5 checkout is missing $name`: $Root"
        }
    }
}

if (Test-Path -LiteralPath $destinationFull) {
    $gitDirectory = Join-Path $destinationFull ".git"
    if (Test-Path -LiteralPath $gitDirectory -PathType Container) {
        $safePath = $destinationFull.Replace("\", "/")
        $resolvedCommit = (
            & git -c "safe.directory=$safePath" -C $destinationFull rev-parse HEAD
        ).Trim()
        if ($LASTEXITCODE -ne 0 -or $resolvedCommit -ne $expectedCommit) {
            throw "Existing YOLOv5 checkout does not match locked commit $expectedCommit"
        }
        $resolvedTag = (
            & git -c "safe.directory=$safePath" -C $destinationFull `
                describe --tags --exact-match HEAD
        ).Trim()
        if ($LASTEXITCODE -ne 0 -or $resolvedTag -ne $expectedTag) {
            throw "Existing YOLOv5 checkout is not locked to tag $expectedTag"
        }
        Assert-RequiredFiles -Root $destinationFull
        Write-TagLock -Root $destinationFull
        Remove-Item -LiteralPath $gitDirectory -Recurse -Force
    } else {
        Assert-RequiredFiles -Root $destinationFull
        $marker = Join-Path $destinationFull $tagFileName
        if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
            throw "Existing non-git YOLOv5 directory has no $tagFileName marker"
        }
        $markerLines = @(Get-Content -LiteralPath $marker -Encoding UTF8)
        $actualTag = $markerLines[0].Trim()
        if ($actualTag -ne $expectedTag) {
            throw "Existing YOLOv5 marker is $actualTag; expected $expectedTag"
        }
        $actualCommit = (
            $markerLines |
                Where-Object { $_.StartsWith("commit=") } |
                Select-Object -First 1
        ) -replace "^commit=", ""
        if ($actualCommit -ne $expectedCommit) {
            throw "Existing YOLOv5 marker commit does not match $expectedCommit"
        }
        Write-TagLock -Root $destinationFull
    }
} else {
    $parent = Split-Path -Parent $destinationFull
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $staging = Join-Path $parent (".yolov5-fetch-" + [guid]::NewGuid().ToString("N"))
    try {
        & git clone --branch $expectedTag --depth 1 $repository $staging
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone YOLOv5 $expectedTag"
        }
        $safePath = $staging.Replace("\", "/")
        $resolvedCommit = (
            & git -c "safe.directory=$safePath" -C $staging rev-parse HEAD
        ).Trim()
        if ($LASTEXITCODE -ne 0 -or $resolvedCommit -ne $expectedCommit) {
            throw "YOLOv5 commit verification failed"
        }
        Assert-RequiredFiles -Root $staging
        Write-TagLock -Root $staging
        Remove-Item -LiteralPath (Join-Path $staging ".git") -Recurse -Force
        Move-Item -LiteralPath $staging -Destination $destinationFull
    } finally {
        if (Test-Path -LiteralPath $staging) {
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

if (Get-ChildItem -LiteralPath $destinationFull -Directory -Force -Recurse |
    Where-Object { $_.Name -eq ".git" }) {
    throw "Sanitized YOLOv5 runtime still contains a .git directory"
}

Write-Host "YOLOv5 $expectedTag runtime is ready: $destinationFull"
