; Inno Setup 6 script — build after PyInstaller (dist\PictureStudio\...).
; Adjust AppId if you fork the product.

#define MyAppName "Picture Studio"
#define MyAppVersion "2.5.2"
#define MyAppPublisher "Picture Studio"
#define MyAppURL "https://github.com/oOMaikOo/Picture"
#define MyAppExeName "PictureStudio.exe"

[Setup]
AppId={{A7C4E2F1-9B8D-4E2A-91F0-PICTURESTUDIO}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\..\dist\installer
OutputBaseFilename=PictureStudio-{#MyAppVersion}-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
; Uncomment when you have a code-signing certificate:
; SignTool=signtool

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Expect PyInstaller onedir output at dist\PictureStudio\
Source: "..\..\dist\PictureStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
