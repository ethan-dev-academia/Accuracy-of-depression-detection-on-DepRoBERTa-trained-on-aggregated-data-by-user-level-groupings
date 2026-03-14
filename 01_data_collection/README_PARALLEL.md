# 🔍 Reddit User Analyzer - Parallel Processing Guide

This guide explains how to run multiple instances of the Reddit User Analyzer simultaneously for faster processing.

## 🚀 Quick Start

### Option 1: Batch Script (Windows)
```batch
run_parallel_analysis.bat
```

### Option 2: PowerShell Script
```powershell
.\run_parallel_analysis.ps1
```

### Option 3: Individual Instance Launcher
```bash
# Terminal 1
python launch_instance.py 1

# Terminal 2  
python launch_instance.py 2
```

## 🔑 API Key Configuration

### Current Setup
The system is configured with **2 API keys** (currently using the same key for both instances):

- **Instance 1**: Uses `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
- **Instance 2**: Uses `REDDIT_CLIENT_ID_2` and `REDDIT_CLIENT_SECRET_2`

### Adding More API Keys
To add more instances, edit `reddit_api_config.py`:

```python
REDDIT_API_KEYS = {
    1: {
        'client_id': 'your_first_client_id',
        'client_secret': 'your_first_client_secret',
        'user_agent': 'RedditUserAnalyzer/1.0-Instance1'
    },
    2: {
        'client_id': 'your_second_client_id',
        'client_secret': 'your_second_client_secret',
        'user_agent': 'RedditUserAnalyzer/1.0-Instance2'
    },
    3: {
        'client_id': 'your_third_client_id',
        'client_secret': 'your_third_client_secret',
        'user_agent': 'RedditUserAnalyzer/1.0-Instance3'
    }
}
```

## 📊 How Parallel Processing Works

### Username Distribution
- **Total usernames**: ~826,965 (from CSV files)
- **Instance 1**: Processes usernames 0 to 413,482
- **Instance 2**: Processes usernames 413,483 to 826,964

### Performance Benefits
- **2x faster processing** with 2 instances
- **Independent rate limits** (60 requests/minute per API key)
- **Automatic checkpoint repair** if files get corrupted
- **Resume capability** for each instance

## 🛠️ Command Line Options

### Basic Usage
```bash
python reddit_user_analyzer.py --instance=1 --instances=2
```

### Advanced Options
```bash
python reddit_user_analyzer.py \
  --instance=1 \
  --instances=2 \
  --threading \
  --workers=8 \
  --batch-size=100 \
  --checkpoint=100
```

### Available Options
- `--instance=N`: Set instance ID (1, 2, 3, etc.)
- `--instances=N`: Set total number of instances
- `--threading`: Use threading (recommended for API calls)
- `--workers=N`: Number of worker threads
- `--batch-size=N`: Usernames per batch
- `--checkpoint=N`: Save checkpoint every N users

## 📁 Output Files

### Instance-Specific Files
- **Instance 1**: `reddit_user_analysis_YYYYMMDD_HHMMSS.json`
- **Instance 2**: `reddit_user_analysis_YYYYMMDD_HHMMSS.json`

### Checkpoint Files
- **Instance 1**: `reddit_user_analysis_checkpoint_YYYYMMDD_HHMMSS.json`
- **Instance 2**: `reddit_user_analysis_checkpoint_YYYYMMDD_HHMMSS.json`

### Content Samples
- **Combined**: `reddit_content_samples_YYYYMMDD_HHMMSS.json`

## 🔧 Automatic Repair Features

The system automatically detects and repairs corrupted checkpoint files:

- **JSON Validation**: Checks file integrity on load
- **Smart Repair**: Finds last complete user object
- **Backup Creation**: Original files backed up before repair
- **Atomic Writes**: Prevents corruption during saving

## 📈 Monitoring Progress

### Real-Time Updates
Each instance shows:
- Current batch progress
- Usernames processed
- Checkpoint saves
- Rate limit status

### Progress Files
Check the AGPacket directory for:
- Current checkpoint files
- Analysis progress
- Error logs

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Each instance has independent rate limits
   - Reduce `--workers` if hitting limits

2. **Memory Issues**
   - Reduce `--batch-size` for large datasets
   - Monitor system memory usage

3. **File Corruption**
   - Automatic repair handles most cases
   - Check `.backup` files if needed

4. **Authentication Errors**
   - Verify API keys in `reddit_api_config.py`
   - Check environment variables

### Performance Tuning

- **Optimal workers**: 8-10 for threading
- **Optimal batch size**: 100-200 for large datasets
- **Checkpoint interval**: 100-200 for large datasets

## 🔄 Resume Capability

### How It Works
- Each instance tracks its own progress
- Checkpoints saved every N users
- Automatic resume from last checkpoint
- Corrupted files automatically repaired

### Manual Resume
```bash
# Resume specific instance
python launch_instance.py 1 --checkpoint=50

# Resume with custom settings
python launch_instance.py 2 --workers=4 --batch-size=50
```

## 📊 Expected Performance

### Processing Speed
- **Sequential**: ~1.68 seconds per username
- **2 Instances**: ~0.84 seconds per username (2x faster)
- **4 Instances**: ~0.42 seconds per username (4x faster)

### Time Estimates
- **Total usernames**: 826,965
- **2 Instances**: ~19 hours
- **4 Instances**: ~9.5 hours
- **8 Instances**: ~4.75 hours

## 🎯 Best Practices

1. **Use threading** for API calls (more reliable than multiprocessing)
2. **Monitor rate limits** and adjust workers accordingly
3. **Regular checkpoints** for large datasets
4. **Separate API keys** for each instance
5. **Monitor system resources** during processing

## 📞 Support

If you encounter issues:
1. Check the automatic repair logs
2. Verify API key configuration
3. Monitor rate limit usage
4. Check system resources

The system is designed to be self-healing and will automatically recover from most common issues.
