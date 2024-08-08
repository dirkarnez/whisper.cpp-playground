@REM run as Administrator
@echo off
cd /d %~dp0

set DOWNLOADS_DIR=%USERPROFILE%\Downloads
set DOWNLOADS_DIR_LINUX=%DOWNLOADS_DIR:\=/%

@REM SET PATH=^
@REM %DOWNLOADS_DIR%\PortableGit\bin;^
@REM %DOWNLOADS_DIR%\x86_64-8.1.0-release-posix-seh-rt_v6-rev0\mingw64;^
@REM %DOWNLOADS_DIR%\x86_64-8.1.0-release-posix-seh-rt_v6-rev0\mingw64\bin;^
@REM %DOWNLOADS_DIR%\cmake-3.26.1-windows-x86_64\bin;

@REM set PATH=^
@REM D:\Softwares\x86_64-8.1.0-release-posix-seh-rt_v6-rev0\mingw64;^
@REM D:\Softwares\x86_64-8.1.0-release-posix-seh-rt_v6-rev0\mingw64\bin;^
@REM D:\Softwares\cmake-3.23.0-rc1-windows-x86_64\bin;

echo %PATH%
cmake.exe -G"MinGW Makefiles" ^
-DCMAKE_BUILD_TYPE=Debug ^
-DCMAKE_EXE_LINKER_FLAGS="-static" ^
-Dwhisper_DIR="%DOWNLOADS_DIR_LINUX%/whisper.cpp/cmake-build/cmakeInstallationPath/lib/cmake/whisper" ^
-DSDL2_DIR="%DOWNLOADS_DIR_LINUX%/SDL/cmake-build/cmakeInstallationPath/lib/cmake/SDL2" ^
-B./cmake-build &&^
cd cmake-build && ( cmake --build . && echo "Successful build" )  &&^
pause
