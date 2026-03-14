# Script to open JupyterLab notebook
cd "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\CheckPoint-11\19"
$pythonPath = "F:\PROCESSING ALGORITHM\2025-ML-NLP-Research\DepRoBERTa-env\Scripts\python.exe"

Write-Host "Starting JupyterLab..."
Start-Process -FilePath $pythonPath -ArgumentList "-m", "jupyterlab", "RESULTS_SECTION_COMPLETE.ipynb" -NoNewWindow

Write-Host "JupyterLab should open in your browser shortly..."
Write-Host "If it doesn't open automatically, navigate to: http://localhost:8888/lab"

