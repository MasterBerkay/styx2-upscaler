@echo off
rem Builds bin\Styx2Trace.addon64: the frame-trace add-on used to analyse the renderer (see NOTES.md).
rem Creating (or rewriting) Styx2Trace.flag next to the game exe writes two frames to Styx2Trace.txt.
setlocal
set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0.."
if not exist bin mkdir bin
if not exist obj mkdir obj
cl /nologo /O2 /MT /EHsc /std:c++17 /W3 /DUNICODE /D_UNICODE /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Ithird_party\reshade\include ^
   /Foobj\ analysis\trace.cpp /LD /Febin\Styx2Trace.addon64 /link /IMPLIB:obj\trace.lib d3d11.lib dxguid.lib user32.lib || exit /b 1
echo Build OK: %CD%\bin\Styx2Trace.addon64
