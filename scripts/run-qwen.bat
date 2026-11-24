@echo off
setlocal EnableExtensions
title NVR - Qwen3.8-27B

REM This script lives in NVR\scripts. Move to the repository root.
cd /d "%~dp0.."
if errorlevel 1 (
    echo [ERROR] Could not enter the NVR repository folder.
    pause
    exit /b 1
)

if not exist "Cargo.toml" (
    echo [ERROR] Cargo.toml was not found. Keep this script inside the NVR\scripts folder.
    pause
    exit /b 1
)

REM Accept a model path as the first argument, or ask for it interactively.
set "MODEL_PATH=%~1"
if not defined MODEL_PATH (
    echo.
    echo Enter the full path to your Qwen3.8-27B GGUF file.
    set /p "MODEL_PATH=GGUF path: "
)
REM Remove quotes pasted as part of the interactive input.
set "MODEL_PATH=%MODEL_PATH:"=%"

if not defined MODEL_PATH (
    echo [ERROR] No model path was provided.
    pause
    exit /b 1
)

if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found:
    echo "%MODEL_PATH%"
    pause
    exit /b 1
)

REM Build NVR on first run if its release executable is missing.
if not exist "target\release\nvr.exe" (
    where cargo >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Rust Cargo was not found. Install Rust stable, reopen this window, and try again.
        pause
        exit /b 1
    )

    echo.
    echo Building NVR in release mode...
    cargo build --release
    if errorlevel 1 (
        echo [ERROR] NVR build failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting Qwen3.8-27B through NVR...
echo Context: 8192 ^| Batch: 512 ^| Physical batch: 512 ^| GPU layers: 999
echo Local endpoint: http://127.0.0.1:1234
echo Keep this window open while using the model.
echo.

"target\release\nvr.exe" run qwen3_8_27b --model "%MODEL_PATH%" --port 1234 --ctx 8192 --batch 512 --ubatch 512 --gpu-layers 999
set "NVR_EXIT=%ERRORLEVEL%"

echo.
if not "%NVR_EXIT%"=="0" (
    echo NVR exited with code %NVR_EXIT%.
) else (
    echo NVR has stopped.
)
pause
exit /b %NVR_EXIT%
