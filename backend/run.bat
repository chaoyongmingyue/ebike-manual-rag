@echo off
cd /d %~dp0
pip install -r requirements.txt
python search_server.py
pause
