@echo off
echo Starting parallel Reddit analysis with 2 instances...
echo.

echo Setting up environment variables for Instance 1...
set REDDIT_CLIENT_ID=foI3sdH3CG5V-JQZN7ymeg
set REDDIT_CLIENT_SECRET=dspoNLvUGgyOdsFmZTMaP2gM-klwjA

echo Setting up environment variables for Instance 2...
set REDDIT_CLIENT_ID_2=foI3sdH3CG5V-JQZN7ymeg
set REDDIT_CLIENT_SECRET_2=dspoNLvUGgyOdsFmZTMaP2gM-klwjA

echo.
echo Starting Instance 1 in new window...
start "Reddit Analyzer - Instance 1" powershell -Command "cd 'F:\PROCESSING ALGORITHM\2025-ML-NLP-Research'; $env:REDDIT_CLIENT_ID='foI3sdH3CG5V-JQZN7ymeg'; $env:REDDIT_CLIENT_SECRET='dspoNLvUGgyOdsFmZTMaP2gM-klwjA'; python reddit_user_analyzer.py --instance=1 --instances=2 --threading --workers=8 --batch-size=100"

echo Starting Instance 2 in new window...
start "Reddit Analyzer - Instance 2" powershell -Command "cd 'F:\PROCESSING ALGORITHM\2025-ML-NLP-Research'; $env:REDDIT_CLIENT_ID_2='foI3sdH3CG5V-JQZN7ymeg'; $env:REDDIT_CLIENT_SECRET_2='dspoNLvUGgyOdsFmZTMaP2gM-klwjA'; python reddit_user_analyzer.py --instance=2 --instances=2 --threading --workers=8 --batch-size=100"

echo.
echo Both instances started! Check the new windows for progress.
echo.
pause
