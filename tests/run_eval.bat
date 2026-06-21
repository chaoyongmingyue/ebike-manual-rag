@echo off
chcp 65001 >nul
cd /d D:\pro_claude\ebike-search\tests

echo ═══════════════════════════════════════════════
echo    Yadea DM6 eBike RAG 自动化评测系统
echo ═══════════════════════════════════════════════
echo.

echo ===== [1/2] Retriever 评测 =====
echo.
python eval_retriever.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ eval_retriever.py 执行失败 (errorlevel=%errorlevel%)
    echo    请确认后端在 localhost:8000 运行中
    pause
    exit /b %errorlevel%
)

echo.
echo ===== [2/2] Answer 评测 =====
echo.
python eval_answer.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ eval_answer.py 执行失败 (errorlevel=%errorlevel%)
    pause
    exit /b %errorlevel%
)

echo.
echo ═══════════════════════════════════════════════
echo    评测完成 — 报告已生成：
echo      tests\retriever_report.md
echo      tests\answer_report.md
echo ═══════════════════════════════════════════════
echo.

rem 快速对照目标
echo ───── 关键指标对照 ─────
findstr /C:"Recall@5" retriever_report.md 2>nul | findstr /V "##"
findstr /C:"MRR" retriever_report.md 2>nul | findstr /V "##\|目标"
findstr /C:"Groundedness" answer_report.md 2>nul | findstr /V "##\|目标"
findstr /C:"Key Fact" answer_report.md 2>nul | findstr /V "##\|目标"
findstr /C:"Rejection" answer_report.md 2>nul | findstr /V "##\|目标"

pause
