@echo off
REM =============================================================================
REM GitHub Repository Initialization Script
REM =============================================================================
REM Project: Game Player Churn Prediction
REM Usage: Run this script to initialize and push to GitHub
REM =============================================================================

echo.
echo =============================================================================
echo  Game Player Churn Prediction - GitHub Setup
echo =============================================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed. Please install Git first.
    echo Download: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Initialize git repository
echo [1/6] Initializing git repository...
git init
echo.

REM Add all files
echo [2/6] Adding files to git...
git add .
echo.

REM Commit
echo [3/6] Creating initial commit...
git commit -m "feat: Complete game player churn prediction project"
echo.

REM Set main branch
echo [4/6] Setting main branch...
git branch -M main
echo.

REM Add remote
echo [5/6] Adding remote repository...
echo Please enter your GitHub username:
set /p USERNAME=
git remote add origin https://github.com/%USERNAME%/game-player-churn-analysis.git
echo.

REM Push
echo [6/6] Pushing to GitHub...
echo This may take a few minutes depending on your network...
echo.
git push -u origin main

echo.
echo =============================================================================
echo  Done! Visit: https://github.com/%USERNAME%/game-player-churn-analysis
echo =============================================================================
echo.
pause
