@echo off
REM Reddit ML Toolkit - Windows Batch Launcher
REM Usage: reddit-toolkit [command] [options]

cd /d "%~dp0"
python reddit_ml_toolkit.py %*
