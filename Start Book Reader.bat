@echo off
title Book Reader
cd /d "%~dp0"
rem The Python environment lives in a short path because Windows chokes on long ones.
set UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.book-reader-env
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo Starting Book Reader (first start can take a minute)...
uv sync --quiet
start "" /b cmd /c "timeout /t 12 /nobreak >nul & start http://localhost:8765"
uv run python app.py
pause
