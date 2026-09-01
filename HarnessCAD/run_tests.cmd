@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%ROOT%..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

cd /d "%ROOT%"
"%PYTHON_EXE%" -m unittest backend.test_harness_v2 -v
if errorlevel 1 exit /b 1

if not exist "%ROOT%frontend\node_modules" (
  echo Installing frontend dependencies...
  pushd "%ROOT%frontend"
  call npm install
  if errorlevel 1 exit /b 1
  popd
)

pushd "%ROOT%frontend"
call npm run build
set "BUILD_STATUS=%ERRORLEVEL%"
popd
exit /b %BUILD_STATUS%
