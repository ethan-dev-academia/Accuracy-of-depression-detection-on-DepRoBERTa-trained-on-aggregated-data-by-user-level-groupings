# Reddit ML Toolkit - Quick Start Guide

## 🚀 Unified Terminal Interface

The Reddit ML Toolkit provides a single command-line interface for all Reddit data operations.

### 📦 What's Included

- **reddit_ml_toolkit.py** - Main unified interface
- **reddit-toolkit.bat** - Windows batch launcher
- **reddit-toolkit.ps1** - PowerShell launcher

### 🔧 Quick Setup

1. **Set API Keys** (First time only):
   ```bash
   python reddit_ml_toolkit.py api --set
   ```

2. **Check System Status**:
   ```bash
   python reddit_ml_toolkit.py status
   ```

### 🚀 Common Commands

#### Data Collection
```bash
# Basic threaded collection
python reddit_ml_toolkit.py collect --threading --workers 8

# Resume from checkpoint
python reddit_ml_toolkit.py collect --resume

# Fresh start
python reddit_ml_toolkit.py collect --fresh
```

#### Parallel Processing
```bash
# Run 2 instances in parallel
python reddit_ml_toolkit.py parallel --instances 2

# Run 4 instances with custom settings
python reddit_ml_toolkit.py parallel --instances 4 --workers 6 --batch-size 50

# Run in background
python reddit_ml_toolkit.py parallel --background
```

#### Data Validation
```bash
# Full ML validation
python reddit_ml_toolkit.py validate

# Quick summary
python reddit_ml_toolkit.py validate --summary
```

#### API Management
```bash
# Show current API config
python reddit_ml_toolkit.py api --show

# Set API keys interactively
python reddit_ml_toolkit.py api --set --instance 1

# Edit config file
python reddit_ml_toolkit.py api --file --edit
```

### 💡 Typical Workflow

1. **Setup**: `python reddit_ml_toolkit.py api --set`
2. **Start Collection**: `python reddit_ml_toolkit.py collect --threading`
3. **Monitor Progress**: `python reddit_ml_toolkit.py status`
4. **Validate Data**: `python reddit_ml_toolkit.py validate --summary`
5. **Scale Up**: `python reddit_ml_toolkit.py parallel --instances 2`

### 🔍 Advanced Features

#### Multiple API Keys
- Configure multiple Reddit API keys for parallel processing
- Automatic load balancing across instances
- Environment variable support

#### Intelligent Checkpointing
- Automatic recovery from corrupted files
- Progress tracking with ETA
- Resume capability

#### ML Readiness Assessment
- Content quality analysis
- Diversity metrics
- Temporal coverage evaluation
- ML use case recommendations

### 📊 File Structure

```
📁 F:/DATA STORAGE/AGPacket/
├── reddit_user_analysis_YYYYMMDD_HHMMSS.json
├── reddit_user_analysis_checkpoint_YYYYMMDD_HHMMSS.json
└── ml_validation_report_YYYYMMDD_HHMMSS.json

📁 Project Directory/
├── reddit_ml_toolkit.py          # Unified interface
├── reddit_user_analyzer.py       # Core analysis engine
├── check_processed_files.py      # ML validation
├── reddit_api_config.py          # API configuration
├── reddit-toolkit.bat            # Windows launcher
└── reddit-toolkit.ps1            # PowerShell launcher
```

### 🎯 Performance Tips

1. **Optimal Threading**: Use 8 workers for most systems
2. **Batch Size**: 100 items per batch works well
3. **Parallel Instances**: 2-4 instances max (API rate limits)
4. **Checkpointing**: Save every 100 users for large datasets

### 🔧 Troubleshooting

#### No API Credentials
```bash
python reddit_ml_toolkit.py api --set
```

#### Corrupted Files
The toolkit automatically detects and repairs corrupted JSON files.

#### Rate Limiting
Use parallel processing with multiple API keys to handle rate limits.

#### Memory Issues
Process data in smaller batches or use checkpointing.

### 📈 ML Readiness Scores

- **90-100**: Excellent for advanced ML
- **70-89**: Good for most ML tasks
- **50-69**: Basic ML applications
- **<50**: Data quality improvement needed

### 🆘 Getting Help

```bash
python reddit_ml_toolkit.py help
```

This shows detailed command documentation and examples.
