#!/usr/bin/env python3
"""
Reddit ML Toolkit - Unified Terminal Interface
A comprehensive command-line tool for Reddit data collection, analysis, and ML validation.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

class RedditMLToolkit:
    """Unified interface for Reddit data operations."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = Path("F:/DATA STORAGE/AGPacket")
        
    def print_banner(self):
        """Print the toolkit banner."""
        print("\n" + "="*80)
        print("🚀 REDDIT ML TOOLKIT - Unified Terminal Interface")
        print("="*80)
        print("📊 Data Collection | 🔍 Analysis | 🤖 ML Validation | ⚡ Parallel Processing")
        print("="*80)
    
    def show_status(self):
        """Show current system status."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        # Check data directory
        if self.data_dir.exists():
            data_files = list(self.data_dir.glob("reddit_user_analysis_*.json"))
            checkpoint_files = list(self.data_dir.glob("*checkpoint*.json"))
            
            print(f"📁 Data directory: {self.data_dir}")
            print(f"📄 Analysis files: {len(data_files)}")
            print(f"💾 Checkpoint files: {len(checkpoint_files)}")
            
            if data_files:
                total_size = sum(f.stat().st_size for f in data_files) / (1024**3)
                print(f"💽 Total data size: {total_size:.2f} GB")
                
                # Show latest file
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                print(f"🕒 Latest file: {latest_file.name}")
                print(f"📅 Last modified: {datetime.fromtimestamp(latest_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"❌ Data directory not found: {self.data_dir}")
        
        # Check API configuration
        has_api_key = bool(os.getenv('REDDIT_CLIENT_ID') or os.getenv('REDDIT_CLIENT_ID_1'))
        has_api_secret = bool(os.getenv('REDDIT_CLIENT_SECRET') or os.getenv('REDDIT_CLIENT_SECRET_1'))
        
        print(f"🔑 API credentials: {'✅ Configured' if has_api_key and has_api_secret else '❌ Missing'}")
        
        # Check script files
        scripts = ['reddit_user_analyzer.py', 'check_processed_files.py', 'reddit_api_config.py']
        for script in scripts:
            exists = (self.base_dir / script).exists()
            print(f"📜 {script}: {'✅ Available' if exists else '❌ Missing'}")
    
    def run_data_collection(self, args):
        """Run Reddit data collection."""
        print("\n🚀 STARTING REDDIT DATA COLLECTION")
        print("-" * 50)
        
        # Build command for reddit_user_analyzer.py
        cmd = [sys.executable, 'reddit_user_analyzer.py']
        
        # Add threading options
        if args.threading:
            cmd.extend(['--threading'])
            if args.workers:
                cmd.extend(['--workers', str(args.workers)])
            if args.batch_size:
                cmd.extend(['--batch-size', str(args.batch_size)])
        
        # Add parallel processing options
        if args.parallel:
            if args.instance:
                cmd.extend(['--instance', str(args.instance)])
            if args.instances:
                cmd.extend(['--instances', str(args.instances)])
        
        # Add checkpoint options
        if args.checkpoint:
            cmd.extend(['--checkpoint', str(args.checkpoint)])
        
        # Add other options
        if args.resume:
            cmd.append('--resume')
        if args.fresh:
            cmd.append('--fresh')
        
        print(f"⚡ Command: {' '.join(cmd)}")
        print("-" * 50)
        
        try:
            subprocess.run(cmd, cwd=self.base_dir)
        except KeyboardInterrupt:
            print("\n⏹️  Data collection interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during data collection: {e}")
    
    def run_parallel_collection(self, args):
        """Run parallel data collection using multiple instances."""
        print("\n⚡ STARTING PARALLEL DATA COLLECTION")
        print("-" * 50)
        
        num_instances = args.instances or 2
        
        # Check for multiple API keys
        api_keys_available = []
        for i in range(1, num_instances + 1):
            client_id_env = f'REDDIT_CLIENT_ID_{i}' if i > 1 else 'REDDIT_CLIENT_ID'
            client_secret_env = f'REDDIT_CLIENT_SECRET_{i}' if i > 1 else 'REDDIT_CLIENT_SECRET'
            
            if os.getenv(client_id_env) and os.getenv(client_secret_env):
                api_keys_available.append(i)
        
        if len(api_keys_available) < num_instances:
            print(f"⚠️  Warning: Only {len(api_keys_available)} API keys configured for {num_instances} instances")
            print("💡 Configure additional API keys in environment variables or reddit_api_config.py")
        
        # Launch instances
        processes = []
        for instance_id in range(1, num_instances + 1):
            cmd = [
                sys.executable, 'reddit_user_analyzer.py',
                '--instance', str(instance_id),
                '--instances', str(num_instances),
                '--threading',
                '--workers', str(args.workers or 8),
                '--batch-size', str(args.batch_size or 100)
            ]
            
            print(f"🚀 Launching instance {instance_id}/{num_instances}...")
            
            try:
                if args.background:
                    # Run in background
                    process = subprocess.Popen(cmd, cwd=self.base_dir)
                    processes.append(process)
                else:
                    # Run in new terminal window (Windows)
                    if sys.platform == "win32":
                        subprocess.Popen([
                            'start', 'cmd', '/k', 
                            f'cd /d "{self.base_dir}" && {" ".join(cmd)}'
                        ], shell=True)
                    else:
                        # Linux/Mac
                        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', 
                                        f'cd "{self.base_dir}" && {" ".join(cmd)}; read'])
            
            except Exception as e:
                print(f"❌ Failed to launch instance {instance_id}: {e}")
            
            time.sleep(2)  # Brief delay between launches
        
        if args.background and processes:
            print(f"\n✅ Launched {len(processes)} background processes")
            print("💡 Use 'ps aux | grep reddit_user_analyzer' to monitor")
        else:
            print(f"\n✅ Launched {num_instances} instances in separate terminals")
    
    def run_validation(self, args):
        """Run ML data validation."""
        print("\n🔍 STARTING ML DATA VALIDATION")
        print("-" * 50)
        
        cmd = [sys.executable, 'check_processed_files.py']
        
        try:
            subprocess.run(cmd, cwd=self.base_dir)
            
            # Show summary if available
            if args.summary:
                print("\n📋 SHOWING VALIDATION SUMMARY")
                print("-" * 50)
                summary_cmd = [sys.executable, 'show_ml_validation_summary.py']
                subprocess.run(summary_cmd, cwd=self.base_dir)
                
        except Exception as e:
            print(f"❌ Error during validation: {e}")
    
    def configure_api(self, args):
        """Configure API keys."""
        print("\n🔑 API CONFIGURATION")
        print("-" * 50)
        
        if args.show:
            # Show current configuration
            print("Current API configuration:")
            for i in range(1, 5):  # Check up to 4 API keys
                client_id_env = f'REDDIT_CLIENT_ID_{i}' if i > 1 else 'REDDIT_CLIENT_ID'
                client_secret_env = f'REDDIT_CLIENT_SECRET_{i}' if i > 1 else 'REDDIT_CLIENT_SECRET'
                
                client_id = os.getenv(client_id_env)
                client_secret = os.getenv(client_secret_env)
                
                if client_id or client_secret:
                    print(f"  Instance {i}:")
                    print(f"    Client ID: {client_id[:10] + '...' if client_id else 'Not set'}")
                    print(f"    Client Secret: {'***' if client_secret else 'Not set'}")
        
        elif args.set:
            # Interactive API key setup
            instance = args.instance or 1
            
            print(f"Setting up API keys for instance {instance}:")
            client_id = input("Enter Reddit Client ID: ").strip()
            client_secret = input("Enter Reddit Client Secret: ").strip()
            
            if client_id and client_secret:
                # Update environment for current session
                client_id_env = f'REDDIT_CLIENT_ID_{instance}' if instance > 1 else 'REDDIT_CLIENT_ID'
                client_secret_env = f'REDDIT_CLIENT_SECRET_{instance}' if instance > 1 else 'REDDIT_CLIENT_SECRET'
                
                os.environ[client_id_env] = client_id
                os.environ[client_secret_env] = client_secret
                
                print(f"✅ API keys set for instance {instance} (current session only)")
                print("💡 For persistent storage, add to reddit_api_config.py or environment variables")
            else:
                print("❌ Both Client ID and Secret are required")
        
        elif args.file:
            # Show or edit reddit_api_config.py
            config_file = self.base_dir / 'reddit_api_config.py'
            if config_file.exists():
                print(f"📄 Configuration file: {config_file}")
                if args.edit:
                    try:
                        os.system(f'notepad "{config_file}"' if sys.platform == "win32" else f'nano "{config_file}"')
                    except:
                        print(f"💡 Manually edit: {config_file}")
                else:
                    print("Use --edit to modify the configuration file")
            else:
                print("❌ reddit_api_config.py not found")
    
    def show_help(self):
        """Show detailed help information."""
        print("\n📖 REDDIT ML TOOLKIT - HELP")
        print("="*60)
        
        help_sections = {
            "🚀 Data Collection": [
                "collect --threading --workers 8 --batch-size 100",
                "collect --parallel --instances 2",
                "collect --resume (continue from last checkpoint)",
                "collect --fresh (start fresh analysis)"
            ],
            "⚡ Parallel Processing": [
                "parallel --instances 2 --workers 8",
                "parallel --background (run in background)",
                "parallel --instances 4 (use 4 parallel instances)"
            ],
            "🔍 Data Validation": [
                "validate (run full ML validation)",
                "validate --summary (show quick summary)",
                "status (show system status)"
            ],
            "🔑 API Configuration": [
                "api --show (display current config)",
                "api --set --instance 1 (set API keys interactively)",
                "api --file --edit (edit config file)"
            ],
            "💡 Common Workflows": [
                "1. Set up API: api --set",
                "2. Start collection: collect --threading",
                "3. Monitor: status",
                "4. Validate data: validate --summary",
                "5. Parallel processing: parallel --instances 2"
            ]
        }
        
        for section, commands in help_sections.items():
            print(f"\n{section}:")
            for cmd in commands:
                print(f"  reddit-toolkit {cmd}")
        
        print(f"\n🔧 Configuration Files:")
        print(f"  • reddit_api_config.py - API key management")
        print(f"  • reddit_user_analyzer.py - Main analysis engine")
        print(f"  • check_processed_files.py - ML validation")
        
        print(f"\n📁 Data Location:")
        print(f"  • {self.data_dir}")
        
        print("\n" + "="*60)

def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Reddit ML Toolkit - Unified Terminal Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Run data collection')
    collect_parser.add_argument('--threading', action='store_true', help='Enable threading')
    collect_parser.add_argument('--workers', type=int, help='Number of worker threads')
    collect_parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    collect_parser.add_argument('--checkpoint', type=int, help='Checkpoint interval')
    collect_parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    collect_parser.add_argument('--instance', type=int, help='Instance ID for parallel processing')
    collect_parser.add_argument('--instances', type=int, help='Total number of instances')
    collect_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    collect_parser.add_argument('--fresh', action='store_true', help='Start fresh analysis')
    
    # Parallel processing command
    parallel_parser = subparsers.add_parser('parallel', help='Run parallel data collection')
    parallel_parser.add_argument('--instances', type=int, default=2, help='Number of parallel instances')
    parallel_parser.add_argument('--workers', type=int, default=8, help='Worker threads per instance')
    parallel_parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parallel_parser.add_argument('--background', action='store_true', help='Run in background')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Run ML data validation')
    validate_parser.add_argument('--summary', action='store_true', help='Show validation summary')
    
    # API configuration command
    api_parser = subparsers.add_parser('api', help='Configure API keys')
    api_parser.add_argument('--show', action='store_true', help='Show current configuration')
    api_parser.add_argument('--set', action='store_true', help='Set API keys interactively')
    api_parser.add_argument('--file', action='store_true', help='Manage config file')
    api_parser.add_argument('--edit', action='store_true', help='Edit configuration file')
    api_parser.add_argument('--instance', type=int, help='API instance number')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show detailed help')
    
    return parser

def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    toolkit = RedditMLToolkit()
    toolkit.print_banner()
    
    if not args.command:
        toolkit.show_status()
        print("\n💡 Use 'python reddit_ml_toolkit.py help' for detailed usage information")
        return
    
    if args.command == 'status':
        toolkit.show_status()
    
    elif args.command == 'collect':
        toolkit.run_data_collection(args)
    
    elif args.command == 'parallel':
        toolkit.run_parallel_collection(args)
    
    elif args.command == 'validate':
        toolkit.run_validation(args)
    
    elif args.command == 'api':
        toolkit.configure_api(args)
    
    elif args.command == 'help':
        toolkit.show_help()
    
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
