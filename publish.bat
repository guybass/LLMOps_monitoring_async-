@echo off
REM ===================================
REM LLMOps Monitoring Build and Publish Script
REM ===================================

echo.
echo ===================================
echo LLMOps Monitoring Build and Publish
echo ===================================
echo.

REM [1/5] Clean old builds
echo [1/5] Cleaning old builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist llamonitor_async.egg-info rmdir /s /q llamonitor_async.egg-info
echo     Done.
echo.

REM [2/5] Verify version in pyproject.toml
echo [2/5] Checking version...
findstr /C:"version" pyproject.toml
echo.
set /p CONFIRM="Is this the correct version to publish? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo.
    echo Publishing cancelled.
    pause
    exit /b 1
)
echo.

REM [3/5] Build package
echo [3/5] Building package...
python -m build
if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo     Done.
echo.

REM [4/5] Check package
echo [4/5] Checking package with twine...
python -m twine check dist/*
if errorlevel 1 (
    echo.
    echo ERROR: Package check failed!
    pause
    exit /b 1
)
echo     Done.
echo.

REM [5/5] Upload to PyPI
echo [5/5] Publishing to PyPI...
echo.
echo IMPORTANT: This will publish to PyPI!
echo.
set /p UPLOAD_CONFIRM="Continue with upload? (y/n): "
if /i not "%UPLOAD_CONFIRM%"=="y" (
    echo.
    echo Publishing cancelled.
    pause
    exit /b 1
)
echo.

python -m twine upload dist/*
if errorlevel 1 (
    echo.
    echo ERROR: Upload failed!
    pause
    exit /b 1
)

echo.
echo ===================================
echo SUCCESS: Package published to PyPI!
echo ===================================
echo.
echo Next steps:
echo   1. Create git tag: git tag v{version}
echo   2. Push tag: git push origin v{version}
echo   3. Create GitHub release
echo.
pause
