#define MyAppName "AI Biaozhu Maintenance 0.2"
#ifndef MyAppVersion
#define MyAppVersion "0.2.3"
#endif
#ifndef MyAppId
  #define MyAppId "{{4C9330ED-77CB-4F81-A467-06B4D6A8FB2B}"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\dist"
#endif
#ifndef MyOutputBaseFilename
  #define MyOutputBaseFilename "AI-Biaozhu-Maintenance-Setup-" + MyAppVersion + "-x64"
#endif
#define MyAppPublisher "AI Biaozhu contributors"
#define MyAppExeName "AI-Biaozhu.exe"
#define MyWorkerExeName "AI-Biaozhu-Worker.exe"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoDescription=Local YOLO dataset annotation studio
DefaultDirName={autopf}\AI-Biaozhu-Maintenance-0.2
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableWelcomePage=no
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir={#MyOutputDir}
OutputBaseFilename={#MyOutputBaseFilename}
Compression=lzma2/normal
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes
Uninstallable=yes
CreateUninstallRegKey=not IsSandboxNoUninstallRegistry
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\build\windows\AI-Biaozhu.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Check: not WizardNoIcons
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon; Check: not WizardNoIcons

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Code]
const
  SandboxNoUninstallRegistryParameter =
    '/AI-BIAOZHU-SANDBOX-NO-UNINSTALL-REGISTRY';

function IsSandboxNoUninstallRegistry: Boolean;
var
  ParameterIndex: Integer;
begin
  Result := False;
  for ParameterIndex := 1 to ParamCount do
  begin
    if CompareText(
      ParamStr(ParameterIndex),
      SandboxNoUninstallRegistryParameter
    ) = 0 then
    begin
      Result := True;
      Exit;
    end;
  end;
end;
