[CmdletBinding()]
param(
    [string]$CondaExe = "",
    [string]$WorkingOutputRoot = "",
    [ValidateRange(1, 32)]
    [int]$NuitkaJobs = 8,
    [switch]$FinalizeExistingNuitka,
    [switch]$ResumeNuitkaBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if ($FinalizeExistingNuitka -and $ResumeNuitkaBuild) {
    throw "FinalizeExistingNuitka and ResumeNuitkaBuild cannot be used together."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectOutputRoot = Join-Path $projectRoot "build\windows"
$outputRoot = if ($WorkingOutputRoot) {
    [System.IO.Path]::GetFullPath($WorkingOutputRoot)
} else {
    $projectOutputRoot
}
$outputRootDrive = [System.IO.Path]::GetPathRoot($outputRoot)
if (
    $WorkingOutputRoot -and
    $outputRoot.TrimEnd("\", "/").Equals(
        $outputRootDrive.TrimEnd("\", "/"),
        [System.StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "WorkingOutputRoot must be a dedicated subdirectory, not a drive root."
}
$nuitkaOutput = Join-Path $outputRoot "nuitka"
$launcherOutput = Join-Path $outputRoot "launchers"
$nuitkaCache = Join-Path $outputRoot "nuitka-cache"
$nuitkaReport = Join-Path $outputRoot "nuitka-report.xml"
$standaloneRoot = Join-Path $projectOutputRoot "AI-Biaozhu.dist"
$sourceImportRoot = Join-Path $projectRoot "src"
$legacyYolo = Join-Path $projectRoot "third_party\runtime\yolov5"
$legacyLockPath = Join-Path $projectRoot "third_party\yolov5.lock.json"
$modelSeedSource = Join-Path $projectRoot "bundled_models\yolo26s.pt"
$modelSeedSha256 = "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B"
$modelSeedSize = 20422725
$tagFileName = ".ai-biaozhu-yolov5-tag"

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

function Assert-PathInsideProject {
    param([Parameter(Mandatory = $true)][string]$Path)

    $candidate = [System.IO.Path]::GetFullPath($Path)
    foreach ($allowedRoot in @($projectRoot, $outputRoot)) {
        $root = [System.IO.Path]::GetFullPath($allowedRoot)
        $prefix = $root.TrimEnd(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        ) + [System.IO.Path]::DirectorySeparatorChar
        if ($candidate.StartsWith(
            $prefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            return
        }
    }
    throw "Build path is outside the project and working output root: $candidate"
}

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $sourceFull = [System.IO.Path]::GetFullPath($Source).TrimEnd("\", "/")
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $sourceFull -Force -Recurse) {
        $relative = $item.FullName.Substring($sourceFull.Length).TrimStart("\", "/")
        $target = Join-Path $Destination $relative
        if ($item.PSIsContainer) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
        } else {
            New-Item -ItemType Directory -Force `
                -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $item.FullName -Destination $target
        }
    }
}

function Find-EntryPoint {
    param(
        [Parameter(Mandatory = $true)][string]$SearchRoot,
        [Parameter(Mandatory = $true)][string]$ExecutableName
    )

    $matches = @(
        Get-ChildItem -LiteralPath $SearchRoot -File -Force -Recurse `
            -Filter $ExecutableName |
            Where-Object { $_.Directory.Name.EndsWith(".dist") }
    )
    if ($matches.Count -ne 1) {
        throw (
            "Expected exactly one $ExecutableName below $SearchRoot, found " +
            "$($matches.Count)."
        )
    }
    return $matches[0]
}

function Find-WindowsManifestTool {
    $onPath = Get-Command mt.exe -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    $kitsBin = "C:\Program Files (x86)\Windows Kits\10\bin"
    if (Test-Path -LiteralPath $kitsBin -PathType Container) {
        $matches = @(
            Get-ChildItem -LiteralPath $kitsBin -File -Recurse -Filter mt.exe |
                Where-Object { $_.Directory.Name -eq "x64" } |
                Sort-Object FullName -Descending
        )
        if ($matches.Count -gt 0) {
            return $matches[0].FullName
        }
    }
    throw "Windows SDK mt.exe was not found; cannot embed PerMonitorV2 manifest."
}

function Set-GuiManifest {
    param([Parameter(Mandatory = $true)][string]$Executable)

    $manifestTool = Find-WindowsManifestTool
    $dpiManifest = Join-Path $projectRoot "packaging\AI-Biaozhu.exe.manifest"
    if (-not (Test-Path -LiteralPath $dpiManifest -PathType Leaf)) {
        throw "PerMonitorV2 manifest is missing: $dpiManifest"
    }
    $existingManifest = Join-Path $outputRoot "AI-Biaozhu.existing.manifest"
    $mergedManifest = Join-Path $outputRoot "AI-Biaozhu.merged.manifest"
    & $manifestTool -nologo "-inputresource:$Executable;#1" "-out:$existingManifest"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not extract the Nuitka executable manifest."
    }
    & $manifestTool -nologo -manifest $existingManifest $dpiManifest `
        "-outputresource:$Executable;#1"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not embed the PerMonitorV2 executable manifest."
    }
    & $manifestTool -nologo "-inputresource:$Executable;#1" "-out:$mergedManifest"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not verify the embedded executable manifest."
    }
    $manifestText = Get-Content -LiteralPath $mergedManifest -Raw -Encoding UTF8
    if (
        $manifestText -notmatch "PerMonitorV2" -or
        $manifestText -notmatch "longPathAware"
    ) {
        throw "The embedded GUI manifest lacks PerMonitorV2 or longPathAware."
    }
}

$cleanCandidates = if ($FinalizeExistingNuitka -or $ResumeNuitkaBuild) {
    @($launcherOutput, $standaloneRoot)
} else {
    @($nuitkaOutput, $launcherOutput, $standaloneRoot)
}
foreach ($candidate in $cleanCandidates) {
    Assert-PathInsideProject -Path $candidate
    if (Test-Path -LiteralPath $candidate) {
        Remove-Item -LiteralPath $candidate -Recurse -Force
    }
}
Assert-PathInsideProject -Path $nuitkaReport
if (
    -not $FinalizeExistingNuitka -and
    -not $ResumeNuitkaBuild -and
    (Test-Path -LiteralPath $nuitkaReport)
) {
    Remove-Item -LiteralPath $nuitkaReport -Force
}
New-Item -ItemType Directory -Force `
    -Path $nuitkaOutput, $launcherOutput, $nuitkaCache | Out-Null
$env:NUITKA_CACHE_DIR = $nuitkaCache
$existingPythonPath = [Environment]::GetEnvironmentVariable(
    "PYTHONPATH",
    [EnvironmentVariableTarget]::Process
)
$env:PYTHONPATH = if ($existingPythonPath) {
    $sourceImportRoot + [System.IO.Path]::PathSeparator + $existingPythonPath
} else {
    $sourceImportRoot
}
$yoloConfig = Join-Path $outputRoot "yolo-config"
New-Item -ItemType Directory -Force -Path $yoloConfig | Out-Null
$env:YOLO_CONFIG_DIR = $yoloConfig

# Validate the official tag/commit and strip .git before bundling.
& (Join-Path $PSScriptRoot "fetch_yolov5.ps1") -Destination $legacyYolo

# Nuitka 4.1.3 normalizes argv[0] with os.path.normcase on Windows but keeps
# the compiled multidist basenames verbatim.  Lower-case launcher basenames
# therefore make dispatch deterministic regardless of the installed EXE's
# display casing.
$guiLauncher = Join-Path $launcherOutput "ai-biaozhu.py"
$workerLauncher = Join-Path $launcherOutput "ai-biaozhu-worker.py"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $guiLauncher,
    "from ai_biaozhu.app import main`nraise SystemExit(main())`n",
    $utf8NoBom
)
[System.IO.File]::WriteAllText(
    $workerLauncher,
    "from ai_biaozhu.workers.main import main`nraise SystemExit(main())`n",
    $utf8NoBom
)

Push-Location $projectRoot
try {
    # ``conda run`` captures output by default and re-encodes it through the
    # active Windows console code page.  That fails when this project lives in
    # a Chinese path, so always keep the child process on its UTF-8 stream.
    $resolvedPackageOutput = & $CondaExe run --no-capture-output --name yolo python -c `
        "import sys, ai_biaozhu; sys.stdout.write(ai_biaozhu.__file__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not import the application source in the build environment."
    }
    $resolvedPackage = ([string]$resolvedPackageOutput).Trim()
    if (-not $resolvedPackage) {
        throw "The build environment returned an empty application source path."
    }
    $sourcePrefix = [System.IO.Path]::GetFullPath($sourceImportRoot).TrimEnd(
        "\", "/"
    ) + [System.IO.Path]::DirectorySeparatorChar
    $resolvedPackagePath = [System.IO.Path]::GetFullPath($resolvedPackage)
    if (-not $resolvedPackagePath.StartsWith(
        $sourcePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw (
            "Nuitka would compile a stale installed package instead of this source tree: " +
            $resolvedPackagePath
        )
    }

    # Nuitka Multidist creates both entry points in one supported standalone
    # dependency tree. "attach" avoids a console for Explorer-launched GUI
    # sessions while preserving redirected stdin/stdout for the QProcess worker.
    $arguments = @(
        "run", "--no-capture-output", "--name", "yolo", "python", "-m", "nuitka",
        "--standalone",
        "--enable-plugin=pyside6",
        "--nofollow-import-to=*.tests",
        "--nofollow-import-to=*.conftest",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=_pytest",
        "--nofollow-import-to=pandas.util._test_decorators",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=IPython.*",
        "--nofollow-import-to=pygments",
        "--nofollow-import-to=pygments.*",
        "--module-parameter=torch-disable-jit=no",
        "--jobs=$NuitkaJobs",
        "--windows-console-mode=attach",
        "--assume-yes-for-downloads",
        "--output-dir=$nuitkaOutput",
        "--report=$nuitkaReport",
        "--include-package=ai_biaozhu",
        "--include-package-data=ai_biaozhu",
        "--include-package-data=ai_biaozhu.ml",
        "--include-package-data=ultralytics",
        "--include-distribution-metadata=ultralytics",
        "--include-distribution-metadata=torch",
        "--include-distribution-metadata=torchvision",
        "--include-distribution-metadata=onnx",
        "--include-distribution-metadata=onnxruntime-gpu",
        "--include-distribution-metadata=onnxsim",
        "--include-distribution-metadata=onnxslim",
        "--include-package=ultralytics",
        "--include-package=torch",
        "--include-package=torchvision",
        "--include-package=albumentations",
        "--include-package=cv2",
        "--include-package=git",
        "--include-package=matplotlib",
        "--include-package=numpy",
        "--include-package=onnx",
        "--include-package=onnxruntime",
        "--include-package=onnxsim",
        "--include-package=onnxslim",
        "--include-package=pandas",
        "--include-package=PIL",
        "--include-package=psutil",
        "--include-package=requests",
        "--include-package=scipy",
        "--include-package=seaborn",
        "--include-package=tensorboard",
        "--include-package=thop",
        "--include-package=tqdm",
        "--include-package=yaml",
        "--main=$guiLauncher",
        "--main=$workerLauncher"
    )
    if ($FinalizeExistingNuitka) {
        if (-not (Test-Path -LiteralPath $nuitkaReport -PathType Leaf)) {
            throw "Cannot finalize because the Nuitka report is missing."
        }
        [xml]$existingReport = Get-Content -LiteralPath $nuitkaReport -Raw
        $reportRoot = $existingReport.'nuitka-compilation-report'
        if (
            [string]$reportRoot.completion -ne "yes" -or
            [string]$reportRoot.mode -ne "standalone"
        ) {
            throw "Cannot finalize an unsuccessful or non-standalone Nuitka build."
        }
        $compiledModuleNames = @(
            $reportRoot.module | ForEach-Object { [string]$_.name }
        )
        if (
            $compiledModuleNames -notcontains "multidist-1-ai-biaozhu" -or
            $compiledModuleNames -notcontains "multidist-2-ai-biaozhu-worker"
        ) {
            throw (
                "Cannot finalize a Nuitka multidist build whose Windows " +
                "entry names were not normalized to lower case."
            )
        }
    } else {
        if ($ResumeNuitkaBuild) {
            Write-Host "Resuming the existing Nuitka build tree: $nuitkaOutput"
        }
        & $CondaExe @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Nuitka multidist standalone build failed."
        }
    }
    if (-not (Test-Path -LiteralPath $nuitkaReport -PathType Leaf)) {
        throw "Nuitka completed without writing its compilation report."
    }

    $compiledGuiEntry = Find-EntryPoint `
        -SearchRoot $nuitkaOutput -ExecutableName "ai-biaozhu.exe"
    $canonicalGuiPath = Join-Path `
        $compiledGuiEntry.Directory.FullName "AI-Biaozhu.exe"
    if (-not $compiledGuiEntry.Name.Equals(
        "AI-Biaozhu.exe",
        [System.StringComparison]::Ordinal
    )) {
        $temporaryGuiPath = Join-Path `
            $compiledGuiEntry.Directory.FullName ".ai-biaozhu-canonical.exe"
        Move-Item -LiteralPath $compiledGuiEntry.FullName `
            -Destination $temporaryGuiPath
        Move-Item -LiteralPath $temporaryGuiPath -Destination $canonicalGuiPath
    }
    $guiEntry = Get-Item -LiteralPath $canonicalGuiPath
    # Multidist produces one binary that dispatches by its invocation name.
    # Nuitka intentionally leaves copying/renaming the additional entry points
    # to the distributor, so create the worker name beside the primary binary.
    $workerPath = Join-Path $guiEntry.Directory.FullName "AI-Biaozhu-Worker.exe"
    if (-not (Test-Path -LiteralPath $workerPath -PathType Leaf)) {
        Copy-Item -LiteralPath $guiEntry.FullName -Destination $workerPath
    }
    $workerEntry = Find-EntryPoint `
        -SearchRoot $nuitkaOutput -ExecutableName "AI-Biaozhu-Worker.exe"
    if ($guiEntry.Directory.FullName -ne $workerEntry.Directory.FullName) {
        throw "Nuitka did not place GUI and worker in the same multidist tree."
    }
    Copy-Tree -Source $guiEntry.Directory.FullName -Destination $standaloneRoot
    Set-GuiManifest -Executable (Join-Path $standaloneRoot "AI-Biaozhu.exe")

    if (-not (Test-Path -LiteralPath $modelSeedSource -PathType Leaf)) {
        throw "Required verified model seed is missing: $modelSeedSource"
    }
    $seedItem = Get-Item -LiteralPath $modelSeedSource
    $seedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelSeedSource).Hash
    if ($seedItem.Length -ne $modelSeedSize -or $seedHash -ne $modelSeedSha256) {
        throw "Bundled yolo26s.pt does not match its locked size/SHA-256."
    }
    $modelSeedRoot = Join-Path $standaloneRoot "model-seed"
    New-Item -ItemType Directory -Force -Path $modelSeedRoot | Out-Null
    Copy-Item -LiteralPath $modelSeedSource `
        -Destination (Join-Path $modelSeedRoot "yolo26s.pt")

    $thirdPartyRoot = Join-Path $standaloneRoot "third_party"
    New-Item -ItemType Directory -Force -Path $thirdPartyRoot | Out-Null
    $bundledYolo = Join-Path $thirdPartyRoot "yolov5"
    Copy-Item -LiteralPath $legacyYolo -Destination $bundledYolo -Recurse -Force
    $bundledGit = Join-Path $bundledYolo ".git"
    if (Test-Path -LiteralPath $bundledGit) {
        Remove-Item -LiteralPath $bundledGit -Recurse -Force
    }
    # GPU smoke tests import the pinned runtime from source and can leave
    # bytecode caches behind. They are neither source nor deployment inputs,
    # so remove them only from this verified standalone staging subtree.
    foreach ($cache in @(
        Get-ChildItem -LiteralPath $bundledYolo -Directory -Force -Recurse |
            Where-Object { $_.Name -eq "__pycache__" }
    )) {
        Assert-PathInsideProject -Path $cache.FullName
        Remove-Item -LiteralPath $cache.FullName -Recurse -Force
    }
    foreach ($bytecode in @(
        Get-ChildItem -LiteralPath $bundledYolo -File -Force -Recurse |
            Where-Object { $_.Extension -in @(".pyc", ".pyo") }
    )) {
        Assert-PathInsideProject -Path $bytecode.FullName
        Remove-Item -LiteralPath $bytecode.FullName -Force
    }
    Copy-Item -LiteralPath $legacyLockPath `
        -Destination (Join-Path $thirdPartyRoot "yolov5.lock.json")
    if (-not (Test-Path -LiteralPath (Join-Path $bundledYolo $tagFileName) `
        -PathType Leaf)) {
        throw "Bundled YOLOv5 runtime has no immutable tag marker."
    }
    if (Get-ChildItem -LiteralPath $bundledYolo -Directory -Force -Recurse |
        Where-Object { $_.Name -eq ".git" }) {
        throw "Bundled YOLOv5 runtime contains a forbidden .git directory."
    }
    if (
        Get-ChildItem -LiteralPath $bundledYolo -Directory -Force -Recurse |
            Where-Object { $_.Name -eq "__pycache__" }
    ) {
        throw "Bundled YOLOv5 runtime contains a forbidden bytecode cache."
    }

    foreach ($document in @("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md")) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $document) `
            -Destination $standaloneRoot
    }
    $runtimeLicenses = Join-Path $standaloneRoot "THIRD_PARTY_LICENSES"
    & $CondaExe run --no-capture-output --name yolo python `
        (Join-Path $PSScriptRoot "collect_runtime_licenses.py") `
        --output $runtimeLicenses `
        --nuitka-report $nuitkaReport
    if ($LASTEXITCODE -ne 0) {
        throw "Collecting runtime dependency licenses failed."
    }
    $pythonPrefixOutput = & $CondaExe run --no-capture-output --name yolo python -c `
        "import sys; sys.stdout.write(sys.prefix)"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the build environment's Python prefix."
    }
    $pythonPrefix = ([string]$pythonPrefixOutput).Trim()
    if (-not $pythonPrefix) {
        throw "The build environment returned an empty Python prefix."
    }
    $pythonLicense = Join-Path $pythonPrefix "LICENSE_PYTHON.txt"
    if (-not (Test-Path -LiteralPath $pythonLicense -PathType Leaf)) {
        throw "The bundled CPython runtime license is missing: $pythonLicense"
    }
    Copy-Item -LiteralPath $pythonLicense `
        -Destination (Join-Path $runtimeLicenses "CPython-3.11-PSF-LICENSE.txt")
    Copy-Item -LiteralPath (Join-Path $bundledYolo "LICENSE") `
        -Destination (Join-Path $runtimeLicenses "YOLOv5-v7.0-GPL-3.0.txt")

    & (Join-Path $standaloneRoot "AI-Biaozhu-Worker.exe") --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Compiled worker failed its --help smoke test."
    }
} finally {
    Pop-Location
}

Write-Host "Standalone multidist build completed: $standaloneRoot"
