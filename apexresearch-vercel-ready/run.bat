@echo off
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python ingest.py papers
start http://localhost:8000
uvicorn main:app --port 8000
