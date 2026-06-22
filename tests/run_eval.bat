@echo off
chcp 65001 >nul
cd /d D:\pro_claude\ebike-search\tests

echo ========================================================
echo    Yadea DM6 eBike RAG Auto Eval
echo ========================================================
echo.

echo ===== [1/2] Retriever Eval =====
echo.
python eval_retriever.py
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] eval_retriever.py errorlevel=%errorlevel%
    echo        Please ensure backend is running on localhost:8000
    pause
    exit /b %errorlevel%
)

echo.
echo ===== [2/2] Answer Eval =====
echo.
python eval_answer.py
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] eval_answer.py errorlevel=%errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo   Evaluation complete. Reports:
echo     tests\retriever_report.md
echo     tests\answer_report.md
echo ========================================================
echo.

echo ----- Key Metrics -----
findstr /C:"Recall@5" retriever_report.md 2>nul | findstr /V "##"
findstr /C:"MRR" retriever_report.md 2>nul | findstr /V "##"
findstr /C:"Groundedness" answer_report.md 2>nul | findstr /V "##"
findstr /C:"Key Fact" answer_report.md 2>nul | findstr /V "##"
findstr /C:"Rejection" answer_report.md 2>nul | findstr /V "##"

pause
