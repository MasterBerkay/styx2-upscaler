@echo off
rem Builds bin\Styx2Upscaler.addon64 (ReShade add-on, x64) with the VS 2022 Build Tools.
rem Dependencies (not part of the repository): third_party\reshade\include, third_party\ngx\include (see README).
setlocal
set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
if not exist bin mkdir bin
if not exist obj mkdir obj
fxc /nologo /T cs_5_0 /E main /O3 /Vn g_MotionCS /Fh obj\motion_cs.h src\shaders\motion.hlsl >nul || exit /b 1
fxc /nologo /T cs_5_0 /E main /O3 /Vn g_SanitizeCS /Fh obj\sanitize_cs.h src\shaders\sanitize.hlsl >nul || exit /b 1
rc /nologo /fo obj\version.res src\version.rc >nul || exit /b 1
cl /nologo /O2 /MT /EHsc /std:c++17 /W3 /DUNICODE /D_UNICODE /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Ithird_party\reshade\include /Ithird_party\ngx\include /Isrc\imgui /Iobj ^
   /Foobj\ src\addon.cpp /LD /Febin\Styx2Upscaler.addon64 /link /IMPLIB:obj\addon.lib dxguid.lib user32.lib advapi32.lib obj\version.res || exit /b 1
echo Build OK: %~dp0bin\Styx2Upscaler.addon64
