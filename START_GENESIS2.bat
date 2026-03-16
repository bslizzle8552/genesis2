@echo off
cd /d %~dp0
start "Genesis2 UI" http://localhost:8000
python -m src.backends.server --host 0.0.0.0 --port 8000
