#!/usr/bin/env python3
"""
Launcher script for individual Reddit analyzer instances.
Usage: python launch_instance.py <instance_id> [additional_args]
"""

import sys
import os
import subprocess
from pathlib import Path

# Import the API configuration
try:
    from reddit_api_config import get_api_key, get_all_instance_ids
except ImportError:
    print("❌ Error: reddit_api_config.py not found!")
    print("Please make sure the configuration file exists.")
    sys.exit(1)

def launch_instance(instance_id, additional_args=None):
    """Launch a specific instance of the Reddit analyzer."""
    
    # Get API key configuration
    config = get_api_key(instance_id)
    if not config:
        print(f"❌ Error: No configuration found for instance {instance_id}")
        print(f"Available instances: {get_all_instance_ids()}")
        return False
    
    # Set environment variables
    os.environ[f'REDDIT_CLIENT_ID_{instance_id if instance_id > 1 else ""}'] = config['client_id']
    os.environ[f'REDDIT_CLIENT_SECRET_{instance_id if instance_id > 1 else ""}'] = config['client_secret']
    
    # Build command
    cmd = [
        sys.executable, 'reddit_user_analyzer.py',
        '--instance', str(instance_id),
        '--instances', str(max(get_all_instance_ids())),
        '--threading',
        '--workers', '8',
        '--batch-size', '100'
    ]
    
    # Add additional arguments if provided
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🚀 Launching Instance {instance_id}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🔑 Using API key: {config['client_id'][:10]}...")
    print(f"⚡ Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Launch the process
        result = subprocess.run(cmd, env=os.environ)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Instance interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error launching instance: {e}")
        return False

def main():
    """Main function to parse arguments and launch instance."""
    
    if len(sys.argv) < 2:
        print("🔍 Reddit Analyzer Instance Launcher")
        print("=" * 40)
        print("Usage: python launch_instance.py <instance_id> [additional_args]")
        print()
        print("Available instances:")
        for instance_id in get_all_instance_ids():
            config = get_api_key(instance_id)
            print(f"  {instance_id}: {config['user_agent']}")
        print()
        print("Examples:")
        print("  python launch_instance.py 1")
        print("  python launch_instance.py 2 --checkpoint=50")
        print("  python launch_instance.py 1 --no-mp")
        return
    
    try:
        instance_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: Invalid instance ID '{sys.argv[1]}'")
        return
    
    # Get additional arguments
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else None
    
    # Launch the instance
    success = launch_instance(instance_id, additional_args)
    
    if success:
        print(f"✅ Instance {instance_id} completed successfully")
    else:
        print(f"❌ Instance {instance_id} failed or was interrupted")

if __name__ == "__main__":
    main()
