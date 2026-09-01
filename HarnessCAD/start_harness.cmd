@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%ROOT%..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if not exist "%ROOT%frontend\node_modules" (
  echo Installing frontend dependencies...
  pushd "%ROOT%frontend"
  call npm install
  if errorlevel 1 exit /b 1
  popd
)

start "HarnessCAD Backend" /D "%ROOT%" "%PYTHON_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
start "HarnessCAD Frontend" /D "%ROOT%frontend" cmd /k "npm run dev"

echo.
echo HarnessCAD: http://localhost:5173/
echo Episode v2: http://localhost:5173/harness-v2.html
echo API docs: http://127.0.0.1:8000/docs
echo.
