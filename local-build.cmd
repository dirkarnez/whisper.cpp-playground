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

set PATH=^
D:\Softwares\PortableGit\bin;^
D:\Softwares\winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1\mingw64;^
D:\Softwares\winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1\mingw64\bin;^
D:\Softwares\cmake-3.29.3-windows-x86_64\bin;

echo %PATH%
cmake.exe -G"MinGW Makefiles" ^
-DCMAKE_BUILD_TYPE=Debug ^
-DCMAKE_EXE_LINKER_FLAGS="-static" ^
-Dwhisper_ROOT="%DOWNLOADS_DIR_LINUX%/whisper.cpp/cmake-build/cmakeInstallationPath" ^
-DSDL2_DIR="%DOWNLOADS_DIR_LINUX%/SDL-release-2.30.6-winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1/lib/cmake/SDL2" ^
-DSDL2_image_DIR="%DOWNLOADS_DIR_LINUX%/SDL_image-release-2.8.2-winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1/lib/cmake/SDL2_image" ^
-DZLIB_ROOT="%DOWNLOADS_DIR_LINUX%/zlib-v1.3.1-winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1" ^
-DZLIB_USE_STATIC_LIBS=ON ^
-Dlibpng16_ROOT="%DOWNLOADS_DIR_LINUX%/libpng-v1.6.43-winlibs-x86_64-posix-seh-gcc-11.2.0-mingw-w64-9.0.0-r1" ^
-B./cmake-build &&^
cd cmake-build && ( cmake --build . && echo "Successful build" )  &&^
pause
