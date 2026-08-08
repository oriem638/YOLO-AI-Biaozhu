[CmdletBinding()]
param(
    [string]$CondaExe = "",
    [string]$EnvironmentName = "yolo",
    [switch]$MaterializeWheelhouse
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$lockRoot = Join-Path $projectRoot "locks"
$wheelhouse = Join-Path $projectRoot "build\wheelhouse-win-64"

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

New-Item -ItemType Directory -Force -Path $lockRoot | Out-Null
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Invoke-CondaText {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $text = (& $CondaExe @Arguments | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed: conda $($Arguments -join ' ')"
    }
    return $text.Replace("`r`n", "`n")
}

$explicit = Invoke-CondaText -Arguments @(
    "list", "--name", $EnvironmentName, "--explicit", "--md5"
)
[System.IO.File]::WriteAllText(
    (Join-Path $lockRoot "conda-win-64-explicit.txt"),
    $explicit,
    $utf8NoBom
)

$condaJson = Invoke-CondaText -Arguments @(
    "list", "--name", $EnvironmentName, "--json"
)
$normalizedConda = $condaJson | ConvertFrom-Json | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
    (Join-Path $lockRoot "conda-win-64-packages.json"),
    $normalizedConda + "`n",
    $utf8NoBom
)

$freeze = Invoke-CondaText -Arguments @(
    "run", "--no-capture-output", "--name", $EnvironmentName,
    "python", "-m", "pip", "freeze", "--all"
)
$sanitizedFreezeLines = New-Object System.Collections.Generic.List[string]
foreach ($rawLine in $freeze -split "`n") {
    $line = $rawLine.Trim()
    if ($line.StartsWith("-e ")) {
        $sanitizedFreezeLines.Add(
            "# Local editable project source omitted; build the application wheel from this tree."
        )
    } elseif ($line -match "^([^ ]+)\s*@\s*file:") {
        $sanitizedFreezeLines.Add(
            "# $($Matches[1]) is supplied by the explicit Conda lock."
        )
    } elseif ($line) {
        $sanitizedFreezeLines.Add($line)
    }
}
[System.IO.File]::WriteAllText(
    (Join-Path $lockRoot "pip-installed-win-64.txt"),
    ($sanitizedFreezeLines -join "`n") + "`n",
    $utf8NoBom
)

$exactLines = New-Object System.Collections.Generic.List[string]
foreach ($rawLine in $freeze -split "`n") {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#") -or $line.StartsWith("-e ")) {
        continue
    }
    # The application itself (and Conda-supplied bootstrap packages) may be
    # reported as an absolute local file URL.  The release script builds the
    # application wheel from this source tree, so local URLs must never leak
    # into a portable third-party dependency lock.
    if ($line -match "\s@\s*file:") {
        continue
    }
    $exactLines.Add($line)
}
$exactPath = Join-Path $lockRoot "requirements-win-64-exact.txt"
[System.IO.File]::WriteAllText(
    $exactPath,
    ($exactLines -join "`n") + "`n",
    $utf8NoBom
)

& $CondaExe run --no-capture-output --name $EnvironmentName python `
    (Join-Path $PSScriptRoot "export_installed_environment.py") `
    --output (Join-Path $lockRoot "python-installed-audit.json")
if ($LASTEXITCODE -ne 0) {
    throw "Exporting the installed Python distribution audit failed."
}

$pipCheck = Invoke-CondaText -Arguments @(
    "run", "--no-capture-output", "--name", $EnvironmentName,
    "python", "-m", "pip", "check"
)
[System.IO.File]::WriteAllText(
    (Join-Path $lockRoot "pip-check.txt"),
    $pipCheck,
    $utf8NoBom
)

if ($MaterializeWheelhouse) {
    $wheelhouseFull = [System.IO.Path]::GetFullPath($wheelhouse)
    $projectPrefix = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd("\", "/") + "\"
    if (-not $wheelhouseFull.StartsWith(
        $projectPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Wheelhouse path escaped the project: $wheelhouseFull"
    }
    if (Test-Path -LiteralPath $wheelhouseFull) {
        Remove-Item -LiteralPath $wheelhouseFull -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $wheelhouseFull | Out-Null
    & $CondaExe run --no-capture-output --name $EnvironmentName python `
        -m pip download --only-binary=:all: --dest $wheelhouseFull `
        --extra-index-url https://download.pytorch.org/whl/cu128 `
        --requirement $exactPath
    if ($LASTEXITCODE -ne 0) {
        throw "Materializing the transitive release wheelhouse failed."
    }
    & $CondaExe run --no-capture-output --name $EnvironmentName python `
        -m pip wheel --no-deps --wheel-dir $wheelhouseFull $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Building the application wheel failed."
    }
    & $CondaExe run --no-capture-output --name $EnvironmentName python `
        (Join-Path $PSScriptRoot "build_wheelhouse_lock.py") `
        --wheelhouse $wheelhouseFull `
        --expected-requirements $exactPath `
        --lock-output (Join-Path $lockRoot "requirements-win-64.lock") `
        --manifest-output (Join-Path $lockRoot "wheelhouse-win-64-manifest.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Generating the wheelhouse hash lock failed."
    }
} else {
    # Never present an artifact hash lock from an older environment as current.
    foreach ($staleArtifact in @(
        (Join-Path $lockRoot "requirements-win-64.lock"),
        (Join-Path $lockRoot "wheelhouse-win-64-manifest.json")
    )) {
        if (Test-Path -LiteralPath $staleArtifact -PathType Leaf) {
            Remove-Item -LiteralPath $staleArtifact -Force
        }
    }
}

$manifestLines = New-Object System.Collections.Generic.List[string]
foreach ($file in Get-ChildItem -LiteralPath $lockRoot -File | Sort-Object Name) {
    if ($file.Name -eq "SHA256SUMS") {
        continue
    }
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifestLines.Add("$hash  $($file.Name)")
}
[System.IO.File]::WriteAllText(
    (Join-Path $lockRoot "SHA256SUMS"),
    ($manifestLines -join "`n") + "`n",
    $utf8NoBom
)

Write-Host "Environment lock export completed: $lockRoot"
