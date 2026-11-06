@echo off
cd /d "%~dp0"
echo Iniciando ProSports AI Dashboard en http://localhost:8501 ...
.\.venv\Scripts\streamlit.exe run src/dashboard.py --server.port 8501 --logger.level error
pause
