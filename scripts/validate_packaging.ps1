[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelSeed = Join-Path $projectRoot "bundled_models\yolo26s.pt"
$modelSeedSha256 = "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B"
$modelSeedSize = 20422725
if (-not (Test-Path -LiteralPath $modelSeed -PathType Leaf)) {
    throw "Verified yolo26s.pt model seed is missing."
}
$seedItem = Get-Item -LiteralPath $modelSeed
$seedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelSeed).Hash
if ($seedItem.Length -ne $modelSeedSize -or $seedHash -ne $modelSeedSha256) {
    throw "Verified yolo26s.pt model seed does not match its locked size/SHA-256."
}
$parseFailures = New-Object System.Collections.Generic.List[string]
foreach ($script in Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.ps1" -File) {
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $script.FullName,
        [ref]$tokens,
        [ref]$errors
    )
    foreach ($parseError in $errors) {
        $parseFailures.Add(
            "$($script.Name):$($parseError.Extent.StartLineNumber): " +
            $parseError.Message
        )
    }
}
if ($parseFailures.Count -gt 0) {
    throw "PowerShell parser errors:`n$($parseFailures -join "`n")"
}

$lockPath = Join-Path $projectRoot "third_party\yolov5.lock.json"
$lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (
    [string]$lock.ref -ne "v7.0" -or
    [string]$lock.commit -notmatch "^[0-9a-f]{40}$" -or
    [string]$lock.license -ne "GPL-3.0-only"
) {
    throw "third_party/yolov5.lock.json is not a valid immutable v7.0 lock."
}

$runtime = Join-Path $projectRoot "third_party\runtime\yolov5"
if (Test-Path -LiteralPath $runtime -PathType Container) {
    if (Get-ChildItem -LiteralPath $runtime -Directory -Force -Recurse |
        Where-Object { $_.Name -eq ".git" }) {
        throw "YOLOv5 runtime contains a forbidden .git directory."
    }
    $marker = Join-Path $runtime ".ai-biaozhu-yolov5-tag"
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "YOLOv5 runtime is missing .ai-biaozhu-yolov5-tag."
    }
    $markerLines = @(Get-Content -LiteralPath $marker -Encoding UTF8)
    $actualTag = $markerLines[0].Trim()
    if ($actualTag -ne [string]$lock.ref) {
        throw "YOLOv5 marker is $actualTag; expected $($lock.ref)."
    }
    $actualCommit = (
        $markerLines |
            Where-Object { $_.StartsWith("commit=") } |
            Select-Object -First 1
    ) -replace "^commit=", ""
    if ($actualCommit -ne [string]$lock.commit) {
        throw "YOLOv5 marker commit does not match $($lock.commit)."
    }
}

$buildScript = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "build_windows.ps1") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "--standalone",
    "--main=`$guiLauncher",
    "--main=`$workerLauncher",
    "--windows-console-mode=attach",
    "NUITKA_CACHE_DIR",
    "PYTHONPATH",
    "stale installed package",
    "YOLO_CONFIG_DIR",
    "--report=`$nuitkaReport",
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
    "--jobs=`$NuitkaJobs",
    '"ai-biaozhu.py"',
    '"ai-biaozhu-worker.py"',
    "FinalizeExistingNuitka",
    "Cannot finalize an unsuccessful or non-standalone Nuitka build.",
    "entry names were not normalized to lower case.",
    "Copy-Item -LiteralPath `$guiEntry.FullName",
    "--nuitka-report `$nuitkaReport",
    "--include-package=onnxslim",
    "--include-package=tensorboard",
    "--include-package=thop",
    "LICENSE_PYTHON.txt",
    "CPython-3.11-PSF-LICENSE.txt",
    "AI-Biaozhu.exe",
    "AI-Biaozhu-Worker.exe",
    ".ai-biaozhu-yolov5-tag",
    "model-seed",
    "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B",
    "forbidden bytecode cache"
)) {
    if (-not $buildScript.Contains($requiredText)) {
        throw "build_windows.ps1 is missing required text: $requiredText"
    }
}
if ($buildScript.Contains("--onefile")) {
    throw "The Windows build must remain standalone, not onefile."
}
if ($buildScript.Contains("--include-package=IPython")) {
    throw "The desktop build must not force the unused IPython package."
}

$workerSource = Get-Content -LiteralPath `
    (Join-Path $projectRoot "src\ai_biaozhu\workers\main.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "_ignore_polars_in_pyside_feature_probe",
    "pyside_feature_dict",
    'feature_table.setdefault("polars", -1)'
)) {
    if (-not $workerSource.Contains($requiredText)) {
        throw "Frozen worker is missing its PySide/Polars compatibility guard."
    }
}

$iss = Get-Content -LiteralPath `
    (Join-Path $projectRoot "packaging\ai_biaozhu.iss") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "[Setup]",
    "[Files]",
    '#define MyAppName "AI Biaozhu Maintenance 0.2"',
    '#define MyAppId "{{4C9330ED-77CB-4F81-A467-06B4D6A8FB2B}"',
    'DefaultDirName={autopf}\AI-Biaozhu-Maintenance-0.2',
    'AI-Biaozhu-Maintenance-Setup-',
    "AppId={#MyAppId}",
    "AI-Biaozhu.dist",
    "compiler:Default.isl",
    "PrivilegesRequired=lowest",
    "AllowNoIcons=yes",
    "UsePreviousAppDir=yes",
    "Uninstallable=yes",
    "CreateUninstallRegKey=not IsSandboxNoUninstallRegistry",
    "/AI-BIAOZHU-SANDBOX-NO-UNINSTALL-REGISTRY",
    "Check: not WizardNoIcons",
    "ArchitecturesAllowed=x64compatible",
    "UninstallDisplayIcon="
)) {
    if (-not $iss.Contains($requiredText)) {
        throw "ai_biaozhu.iss is missing required text: $requiredText"
    }
}
foreach ($forbiddenText in @(
    '#define MyAppId "{{147EE884-BF66-4DFE-BF0D-2D275C1A62AB}"',
    '#define MyAppName "AI Biaozhu"'
)) {
    if ($iss.Contains($forbiddenText)) {
        throw "Maintenance installer retained an original-edition identity: $forbiddenText"
    }
}
if ($iss -match '(?m)^DefaultDirName=\{autopf\}\\AI-Biaozhu\s*$') {
    throw "Maintenance installer retained the original default install directory."
}
if (-not $iss.Contains("x64compatible")) {
    throw "The installer script must retain the Inno Setup 6.3+ capability token."
}
if ($iss.Contains("[UninstallDelete]")) {
    throw "The installer must not recursively delete user/runtime data on uninstall."
}
if (
    ([regex]::Matches(
        $iss,
        [regex]::Escape("Check: not WizardNoIcons")
    )).Count -ne 2
) {
    throw "Both installer shortcuts must explicitly honor /NOICONS."
}
$installerScript = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "build_installer.ps1") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "AI_BIAOZHU_ISCC",
    "unins000.exe",
    "x64compatible script",
    "CPython-3.11-PSF-LICENSE.txt",
    "model weights or exported model artifacts",
    "model-seed\yolo26s.pt",
    "locked size/SHA-256"
)) {
    if (-not $installerScript.Contains($requiredText)) {
        throw "build_installer.ps1 is missing version detection text: $requiredText"
    }
}
$releaseScript = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "prepare_release.ps1") -Raw -Encoding UTF8
$releasePython = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "prepare_release.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "--source-mirror",
    "--clean-output",
    "--allow-partial-installer-validation",
    "prepare_release.py"
)) {
    if (-not $releaseScript.Contains($requiredText)) {
        throw "prepare_release.ps1 is missing release flow text: $requiredText"
    }
}
foreach ($requiredText in @(
    "FORBIDDEN_SOURCE_SUFFIXES",
    "FORBIDDEN_RUNTIME_SUFFIXES",
    "RUNTIME_MODEL_SEED",
    "MODEL_SEED_SHA256",
    ".ai-biaozhu-maintenance-source-mirror.json",
    'RELEASE_STEM = "AI-Biaozhu-Maintenance"',
    'RELEASE_APPLICATION = "AI Biaozhu Maintenance 0.2"',
    "RELEASE_CHECKSUM_NAME",
    "_verify_zip_entries",
    "_verify_release_evidence",
    "_verify_nuitka_report",
    "_invalidate_release_metadata",
    "allow_partial_installer_validation",
    "partial_sandbox_validation",
    '"status": release_status',
    "_release_completion_state",
    '"YOLOv5n"',
    '"YOLOv8n"',
    '"YOLO11n"',
    '"YOLO26n"',
    'installer_status != "passed"',
    '"partial" if partial_installer else "passed"',
    "nuitka-report.xml",
    "SHA256SUMS"
)) {
    if (-not $releasePython.Contains($requiredText)) {
        throw "prepare_release.py is missing release gate text: $requiredText"
    }
}

$installerSmoke = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "run_installer_smoke.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    'MODEL_SEED_RELATIVE = "model-seed/yolo26s.pt"',
    "MODEL_SEED_SHA256",
    "_verify_model_seed"
)) {
    if (-not $installerSmoke.Contains($requiredText)) {
        throw "run_installer_smoke.py is missing model-seed gate text: $requiredText"
    }
}

$appPathsSource = Get-Content -LiteralPath `
    (Join-Path $projectRoot "src\ai_biaozhu\app_paths.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    'APP_NAME = "',
    '-0.2"',
    'APP_AUTHOR = "AI-Biaozhu-Maintenance"'
)) {
    if (-not $appPathsSource.Contains($requiredText)) {
        throw "Maintenance runtime path identity is missing: $requiredText"
    }
}
$appEntrySource = Get-Content -LiteralPath `
    (Join-Path $projectRoot "src\ai_biaozhu\app.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    'setOrganizationName("AI-Biaozhu-Maintenance")',
    'setApplicationName(',
    '{__version__}'
)) {
    if (-not $appEntrySource.Contains($requiredText)) {
        throw "Maintenance Qt identity is missing: $requiredText"
    }
}

$cudaSmoke = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "run_standalone_cuda_smoke.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "worker_sha256",
    "checkpoint_sha256",
    "expected_job_id",
    "protocol event sequence is not strictly increasing",
    "checkpoint artifact escaped this smoke run",
    '"device": 0'
)) {
    if (-not $cudaSmoke.Contains($requiredText)) {
        throw "CUDA standalone smoke is missing a fail-closed gate: $requiredText"
    }
}

$installerSmoke = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "run_installer_smoke.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "_registered_installations",
    "refusing to replace an existing matching installation",
    "--sandbox-no-uninstall-registry",
    "SANDBOX_NO_UNINSTALL_REGISTRY_PARAMETER",
    "partial_sandbox_validation",
    "not_verified_sandbox_mode",
    "shortcuts_unchanged",
    "installer_sha256",
    "standalone_tree_sha256",
    "installed_tree_sha256",
    "installation directory remains after uninstall",
    "uninstall registration remains"
)) {
    if (-not $installerSmoke.Contains($requiredText)) {
        throw "Installer smoke is missing a safety/integrity gate: $requiredText"
    }
}

$onnxSmoke = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "run_real_onnx_gate_smoke.py") -Raw -Encoding UTF8
foreach ($requiredText in @(
    "checkpoint_sha256",
    "device_validation",
    "protocol contains a foreign job_id",
    "protocol sequence is invalid"
)) {
    if (-not $onnxSmoke.Contains($requiredText)) {
        throw "Real ONNX smoke is missing an evidence-integrity gate: $requiredText"
    }
}

$constraints = Get-Content -LiteralPath `
    (Join-Path $projectRoot "constraints-ml.txt") -Raw -Encoding UTF8
$environment = Get-Content -LiteralPath `
    (Join-Path $projectRoot "environment.yml") -Raw -Encoding UTF8
$pyproject = Get-Content -LiteralPath `
    (Join-Path $projectRoot "pyproject.toml") -Raw -Encoding UTF8
foreach ($pin in @(
    "torch==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
    "PySide6==6.9.2",
    "ultralytics==8.4.82",
    "Nuitka==4.1.3"
)) {
    if (-not ($constraints.Contains($pin) -or $pyproject.Contains($pin))) {
        throw "Required dependency pin is missing: $pin"
    }
}
foreach ($pin in @("torch==2.11.0+cu128", "torchvision==0.26.0+cu128")) {
    if (-not $environment.Contains($pin)) {
        throw "environment.yml is missing required dependency pin: $pin"
    }
}
if ($environment.Contains("-e .")) {
    throw "environment.yml must install the project non-editably for isolation."
}
if (-not $environment.Contains("- .[ml,dev,build]")) {
    throw "environment.yml is missing the non-editable project installation."
}
$lockExportScript = Get-Content -LiteralPath `
    (Join-Path $PSScriptRoot "export_yolo_lock.ps1") -Raw -Encoding UTF8
if (-not $lockExportScript.Contains(
    "--extra-index-url https://download.pytorch.org/whl/cu128"
)) {
    throw "The wheelhouse lock export is missing the PyTorch CUDA 12.8 index."
}
foreach ($requiredText in @(
    'if ($line -match "\s@\s*file:")',
    "Never present an artifact hash lock from an older environment as current"
)) {
    if (-not $lockExportScript.Contains($requiredText)) {
        throw "The lock export is missing portable-lock protection: $requiredText"
    }
}

Write-Host "Packaging static validation passed."
