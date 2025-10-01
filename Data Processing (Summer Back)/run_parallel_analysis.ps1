# PowerShell script to run parallel Reddit analysis
Write-Host "Starting parallel Reddit analysis with 2 instances..." -ForegroundColor Green
Write-Host ""

# Set environment variables for Instance 1
$env:REDDIT_CLIENT_ID = "foI3sdH3CG5V-JQZN7ymeg"
$env:REDDIT_CLIENT_SECRET = "dspoNLvUGgyOdsFmZTMaP2gM-klwjA"

# Set environment variables for Instance 2
$env:REDDIT_CLIENT_ID_2 = "foI3sdH3CG5V-JQZN7ymeg"
$env:REDDIT_CLIENT_SECRET_2 = "dspoNLvUGgyOdsFmZTMaP2gM-klwjA"

Write-Host "Environment variables set for both instances" -ForegroundColor Yellow
Write-Host ""

# Start Instance 1 in new window
Write-Host "Starting Instance 1 in new window..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'F:\PROCESSING ALGORITHM\2025-ML-NLP-Research'; `$env:REDDIT_CLIENT_ID='foI3sdH3CG5V-JQZN7ymeg'; `$env:REDDIT_CLIENT_SECRET='dspoNLvUGgyOdsFmZTMaP2gM-klwjA'; python reddit_user_analyzer.py --instance=1 --instances=2 --threading --workers=8 --batch-size=100" -WindowStyle Normal

# Start Instance 2 in new window
Write-Host "Starting Instance 2 in new window..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'F:\PROCESSING ALGORITHM\2025-ML-NLP-Research'; `$env:REDDIT_CLIENT_ID_2='foI3sdH3CG5V-JQZN7ymeg'; `$env:REDDIT_CLIENT_SECRET_2='dspoNLvUGgyOdsFmZTMaP2gM-klwjA'; python reddit_user_analyzer.py --instance=2 --instances=2 --threading --workers=8 --batch-size=100" -WindowStyle Normal

Write-Host ""
Write-Host "Both instances started! Check the new windows for progress." -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit this launcher..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
