#!/usr/bin/env python3
"""
Reddit User Analyzer - Analyzes usernames from CSV datasets using Reddit API
Gathers post/comment statistics for each unique username found in the data.
Now includes content extraction and grouping functionality.
"""

import pandas as pd
import os
import sys
import time
import json
from pathlib import Path
from collections import Counter, defaultdict
import requests
from datetime import datetime

class RedditUserAnalyzer:
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize the Reddit API analyzer.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret  
            user_agent (str): User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent or "RedditUserAnalyzer/1.0"
        self.access_token = None
        self.rate_limit_remaining = 60  # Reddit allows 60 requests per minute
        self.rate_limit_reset = time.time() + 60
        
    def authenticate(self):
        """Authenticate with Reddit API using client credentials."""
        if not self.client_id or not self.client_secret:
            print("❌ Error: Reddit API credentials not provided!")
            print("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")
            print("Or provide them when initializing the class.")
            return False
            
        try:
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                'grant_type': 'client_credentials'
            }
            auth_headers = {
                'User-Agent': self.user_agent
            }
            
            response = requests.post(
                auth_url,
                data=auth_data,
                headers=auth_headers,
                auth=(self.client_id, self.client_secret)
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                print("✅ Successfully authenticated with Reddit API")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def get_user_info(self, username):
        """
        Get user information from Reddit API.
        
        Args:
            username (str): Reddit username
            
        Returns:
            dict: User information or None if error
        """
        if not self.access_token:
            print("❌ Not authenticated. Please call authenticate() first.")
            return None
            
        # Rate limiting
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - time.time()
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            self.rate_limit_remaining = 60
            self.rate_limit_reset = time.time() + 60
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'User-Agent': self.user_agent
            }
            
            # Get user overview (posts and comments)
            url = f"https://oauth.reddit.com/user/{username}/overview.json?limit=100"
            response = requests.get(url, headers=headers)
            
            self.rate_limit_remaining -= 1
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_user_data(data, username)
            elif response.status_code == 404:
                return {
                    'username': username,
                    'exists': False,
                    'error': 'User not found'
                }
            else:
                return {
                    'username': username,
                    'exists': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            return {
                'username': username,
                'exists': False,
                'error': str(e)
            }
    
    def _parse_user_data(self, data, username):
        """Parse Reddit API response to extract user statistics."""
        try:
            posts = []
            comments = []
            
            if 'data' in data and 'children' in data['data']:
                for child in data['data']['children']:
                    if 'data' in child:
                        post_data = child['data']
                        if 'is_self' in post_data:  # This is a post
                            posts.append({
                                'title': post_data.get('title', ''),
                                'subreddit': post_data.get('subreddit', ''),
                                'score': post_data.get('score', 0),
                                'created_utc': post_data.get('created_utc', 0),
                                'content': post_data.get('selftext', '')  # Add content for posts
                            })
                        else:  # This is a comment
                            comments.append({
                                'body': post_data.get('body', '')[:100] + '...' if len(post_data.get('body', '')) > 100 else post_data.get('body', ''),
                                'subreddit': post_data.get('subreddit', ''),
                                'score': post_data.get('score', 0),
                                'created_utc': post_data.get('created_utc', 0),
                                'content': post_data.get('body', '')  # Add content for comments
                            })
            
            return {
                'username': username,
                'exists': True,
                'total_posts': len(posts),
                'total_comments': len(comments),
                'posts': posts,
                'comments': comments,
                'last_activity': max([p['created_utc'] for p in posts + comments]) if posts or comments else None
            }
            
        except Exception as e:
            return {
                'username': username,
                'exists': False,
                'error': f'Parse error: {str(e)}'
            }
    
    def check_content_duplicates(self, content_samples, existing_database_file=None):
        """
        Check for duplicate content against existing database.
        
        Args:
            content_samples (dict): New content samples to check
            existing_database_file (str): Path to existing database JSON file
            
        Returns:
            dict: Content samples with duplicates removed
        """
        if not existing_database_file:
            print("⚠️  No existing database file provided for duplicate checking")
            return content_samples
        
        try:
            # Load existing database
            with open(existing_database_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            print(f"🔍 Checking for duplicates against existing database: {existing_database_file}")
            
            # Extract all existing content for comparison
            existing_content = set()
            
            # Check if it's the new format (organized structure)
            if 'users' in existing_data:
                for username, user_data in existing_data['users'].items():
                    if 'samples' in user_data:
                        for sample in user_data['samples']:
                            content = sample.get('content', '').strip()
                            if content:
                                existing_content.add(content.lower())
            else:
                # Check if it's the old format (flat list)
                for item in existing_data:
                    if isinstance(item, dict) and 'content' in item:
                        content = item['content'].strip()
                        if content:
                            existing_content.add(content.lower())
            
            print(f"  Found {len(existing_content)} existing content items in database")
            
            # Filter out duplicates
            filtered_content = {}
            total_duplicates = 0
            
            for username, samples in content_samples.items():
                filtered_samples = []
                
                for sample in samples:
                    content = sample.get('content', '').strip()
                    if content and content.lower() not in existing_content:
                        filtered_samples.append(sample)
                    else:
                        total_duplicates += 1
                
                if filtered_samples:
                    filtered_content[username] = filtered_samples
            
            print(f"  Removed {total_duplicates} duplicate content items")
            print(f"  Kept {sum(len(samples) for samples in filtered_content.values())} unique content items")
            
            return filtered_content
            
        except FileNotFoundError:
            print(f"⚠️  Existing database file not found: {existing_database_file}")
            return content_samples
        except Exception as e:
            print(f"❌ Error checking duplicates: {str(e)}")
            return content_samples

    def remove_user_internal_duplicates(self, content_samples):
        """
        Remove duplicate content within the same user's samples.
        
        Args:
            content_samples (dict): Content samples grouped by username
            
        Returns:
            dict: Content samples with internal duplicates removed
        """
        print("🔍 Removing internal duplicates within user samples...")
        
        cleaned_content = {}
        total_internal_duplicates = 0
        
        for username, samples in content_samples.items():
            seen_content = set()
            unique_samples = []
            
            for sample in samples:
                content = sample.get('content', '').strip().lower()
                if content and content not in seen_content:
                    unique_samples.append(sample)
                    seen_content.add(content)
                else:
                    total_internal_duplicates += 1
            
            if unique_samples:
                cleaned_content[username] = unique_samples
        
        print(f"  Removed {total_internal_duplicates} internal duplicate content items")
        print(f"  Kept {sum(len(samples) for samples in cleaned_content.values())} unique content items")
        
        return cleaned_content

    def extract_user_content_samples(self, results, min_words=5, max_words=500, max_samples_per_user=5, existing_database_file=None):
        """
        Extract content samples from user posts/comments and group by user.
        Now includes duplicate detection against existing database and internal duplicates.
        
        Args:
            results (list): List of user analysis results
            min_words (int): Minimum word count for content
            max_words (int): Maximum word count for content
            max_samples_per_user (int): Maximum samples per user
            existing_database_file (str): Path to existing database for duplicate checking
            
        Returns:
            dict: Grouped content samples by username (duplicates removed)
        """
        print(f"\n📝 Extracting content samples ({min_words}-{max_words} words, max {max_samples_per_user} per user)...")
        
        user_content = defaultdict(list)
        
        for user_result in results:
            if not user_result['exists']:
                continue
                
            username = user_result['username']
            content_items = []
            
            # Process posts
            for post in user_result.get('posts', []):
                content = post.get('content', '')
                if content:
                    word_count = len(content.split())
                    if min_words <= word_count <= max_words:
                        content_items.append({
                            'type': 'post',
                            'title': post.get('title', ''),
                            'subreddit': post.get('subreddit', ''),
                            'content': content,
                            'word_count': word_count,
                            'score': post.get('score', 0),
                            'created_utc': post.get('created_utc', 0)
                        })
            
            # Process comments
            for comment in user_result.get('comments', []):
                content = comment.get('content', '')
                if content:
                    word_count = len(content.split())
                    if min_words <= word_count <= max_words:
                        content_items.append({
                            'type': 'comment',
                            'subreddit': comment.get('subreddit', ''),
                            'content': content,
                            'word_count': word_count,
                            'score': comment.get('score', 0),
                            'created_utc': comment.get('created_utc', 0)
                        })
            
            # Sort by score (highest first) and take top samples
            content_items.sort(key=lambda x: x['score'], reverse=True)
            user_content[username] = content_items[:max_samples_per_user]
        
        # Remove internal duplicates within each user's samples
        user_content = self.remove_user_internal_duplicates(user_content)
        
        # Check for duplicates against existing database
        if existing_database_file:
            user_content = self.check_content_duplicates(user_content, existing_database_file)
        
        return dict(user_content)
    
    def save_content_samples(self, content_samples, output_file=None):
        """
        Save content samples to a JSON file with organized structure.
        
        Args:
            content_samples (dict): Content samples grouped by username
            output_file (str): Output file path
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"F:/DATA STORAGE/AGPacket/reddit_content_samples_{timestamp}.json"
        
        # Create organized structure
        organized_content = {
            'metadata': {
                'total_users': len(content_samples),
                'extraction_date': datetime.now().isoformat(),
                'description': 'Reddit content samples grouped by user (5-500 words)'
            },
            'users': {}
        }
        
        for username, samples in content_samples.items():
            organized_content['users'][username] = {
                'sample_count': len(samples),
                'total_words': sum(s['word_count'] for s in samples),
                'samples': samples
            }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(organized_content, f, indent=2, ensure_ascii=False)
            print(f"💾 Content samples saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error saving content samples: {str(e)}")
            return None
    
    def print_content_summary(self, content_samples):
        """
        Print a summary of extracted content samples.
        
        Args:
            content_samples (dict): Content samples grouped by username
        """
        if not content_samples:
            print("No content samples found.")
            return
        
        print(f"\n📊 CONTENT SAMPLES SUMMARY")
        print("=" * 60)
        
        total_users = len(content_samples)
        total_samples = sum(len(samples) for samples in content_samples.values())
        total_words = sum(sum(s['word_count'] for s in samples) for samples in content_samples.values())
        
        print(f"Users with content samples: {total_users}")
        print(f"Total content samples: {total_samples}")
        print(f"Total words: {total_words:,}")
        print(f"Average samples per user: {total_samples/total_users:.1f}")
        print(f"Average words per sample: {total_words/total_samples:.1f}")
        
        # Show top users by sample count
        users_by_samples = sorted(content_samples.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n🏆 Top 10 Users by Sample Count:")
        for i, (username, samples) in enumerate(users_by_samples[:10], 1):
            total_words = sum(s['word_count'] for s in samples)
            print(f"  {i}. {username}: {len(samples)} samples, {total_words:,} words")
        
        # Show content type distribution
        post_count = sum(1 for samples in content_samples.values() for s in samples if s['type'] == 'post')
        comment_count = sum(1 for samples in content_samples.values() for s in samples if s['type'] == 'comment')
        print(f"\n📝 Content Type Distribution:")
        print(f"  Posts: {post_count}")
        print(f"  Comments: {comment_count}")
    
    def get_all_csv_usernames(self, data_dir="F:/DATA STORAGE/RMH Dataset"):
        """
        Extract all unique usernames from CSV files in the data directory.
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            list: List of unique usernames
        """
        usernames = set()
        csv_files = []
        
        data_path = Path(data_dir)
        if data_path.exists():
            csv_files = list(data_path.rglob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {data_dir} directory")
            return []
        
        print(f"Found {len(csv_files)} CSV file(s)")
        
        for csv_file in csv_files:
            try:
                print(f"Reading {csv_file.name}...")
                df = pd.read_csv(csv_file)
                
                if len(df.columns) >= 2:
                    second_col = df.columns[1]
                    # Get usernames from second column, remove NaN and convert to string
                    file_usernames = df[second_col].dropna().astype(str)
                    usernames.update(file_usernames)
                    print(f"  Found {len(file_usernames)} usernames in {csv_file.name}")
                else:
                    print(f"  Warning: {csv_file.name} has fewer than 2 columns")
                    
            except Exception as e:
                print(f"  Error reading {csv_file.name}: {str(e)}")
        
        unique_usernames = sorted(list(usernames))
        print(f"\nTotal unique usernames found: {len(unique_usernames)}")
        return unique_usernames
    
    def analyze_users(self, usernames, output_file=None, delay=1):
        """
        Analyze a list of usernames using Reddit API.
        
        Args:
            usernames (list): List of usernames to analyze
            output_file (str): Optional output file for results
            delay (float): Delay between API requests in seconds
            
        Returns:
            list: List of user analysis results
        """
        if not self.access_token:
            print("❌ Not authenticated. Please call authenticate() first.")
            return []
        
        results = []
        total = len(usernames)
        
        print(f"\n🔍 Analyzing {total} usernames...")
        print("=" * 60)
        
        for i, username in enumerate(usernames, 1):
            print(f"[{i}/{total}] Analyzing: {username}")
            
            user_info = self.get_user_info(username)
            results.append(user_info)
            
            # Print summary
            if user_info['exists']:
                print(f"  ✅ Found: {user_info['total_posts']} posts, {user_info['total_comments']} comments")
            else:
                print(f"  ❌ {user_info['error']}")
            
            # Save progress periodically
            if output_file and i % 10 == 0:
                self._save_results(results, output_file)
            
            # Rate limiting delay
            if i < total:  # Don't delay after the last request
                time.sleep(delay)
        
        # Save final results
        if output_file:
            self._save_results(results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def _save_results(self, results, output_file):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    
    def generate_summary_report(self, results):
        """Generate a summary report of the analysis."""
        if not results:
            return
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY REPORT")
        print("=" * 60)
        
        total_users = len(results)
        existing_users = [r for r in results if r['exists']]
        non_existing_users = [r for r in results if not r['exists']]
        
        print(f"Total usernames analyzed: {total_users}")
        print(f"Existing users: {len(existing_users)}")
        print(f"Non-existing users: {len(non_existing_users)}")
        
        if existing_users:
            total_posts = sum(r['total_posts'] for r in existing_users)
            total_comments = sum(r['total_comments'] for r in existing_users)
            
            print(f"\n📈 Activity Statistics:")
            print(f"  Total posts: {total_posts:,}")
            print(f"  Total comments: {total_comments:,}")
            print(f"  Average posts per user: {total_posts/len(existing_users):.1f}")
            print(f"  Average comments per user: {total_comments/len(existing_users):.1f}")
            
            # Top active users
            active_users = sorted(existing_users, key=lambda x: x['total_posts'] + x['total_comments'], reverse=True)
            print(f"\n🏆 Top 10 Most Active Users:")
            for i, user in enumerate(active_users[:10], 1):
                total_activity = user['total_posts'] + user['total_comments']
                print(f"  {i}. {user['username']}: {user['total_posts']} posts, {user['total_comments']} comments (Total: {total_activity})")
        
        if non_existing_users:
            print(f"\n❌ Non-existing users ({len(non_existing_users)}):")
            for user in non_existing_users[:10]:  # Show first 10
                print(f"  • {user['username']}: {user['error']}")
            if len(non_existing_users) > 10:
                print(f"  ... and {len(non_existing_users) - 10} more")

    def find_last_processed_username(self, agpacket_dir="F:/DATA STORAGE/AGPacket"):
        """
        Find the last processed username from existing analysis files.
        
        Args:
            agpacket_dir (str): Path to AGPacket directory
            
        Returns:
            tuple: (last_username, last_file_path, total_processed) or (None, None, 0)
        """
        try:
            agpacket_path = Path(agpacket_dir)
            if not agpacket_path.exists():
                return None, None, 0
            
            # Look for the most recent user analysis file
            analysis_files = list(agpacket_path.glob("reddit_user_analysis_*.json"))
            if not analysis_files:
                return None, None, 0
            
            # Get the most recent file
            most_recent_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            print(f"🔍 Checking last analysis file: {most_recent_file.name}")
            
            # Load the file and find the last processed username
            with open(most_recent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return None, most_recent_file, 0
            
            # Find the last username in the list
            last_username = None
            total_processed = 0
            
            for item in data:
                if isinstance(item, dict) and item.get('exists') is not None:
                    last_username = item.get('username')
                    total_processed += 1
            
            if last_username:
                print(f"  Last processed username: {last_username}")
                print(f"  Total users processed: {total_processed}")
                return last_username, most_recent_file, total_processed
            else:
                return None, most_recent_file, 0
                
        except Exception as e:
            print(f"❌ Error finding last processed username: {str(e)}")
            return None, None, 0

    def get_resume_usernames(self, all_usernames, last_username=None, agpacket_dir="F:/DATA STORAGE/AGPacket"):
        """
        Get the list of usernames to process, starting from where we left off.
        
        Args:
            all_usernames (list): Complete list of usernames to process
            last_username (str): Last username that was processed
            agpacket_dir (str): Path to AGPacket directory
            
        Returns:
            tuple: (usernames_to_process, already_processed_count, resume_info)
        """
        if not last_username:
            print("🆕 Starting fresh analysis - no previous progress found")
            return all_usernames, 0, {"resuming": False}
        
        try:
            # Find the index of the last processed username
            try:
                last_index = all_usernames.index(last_username)
                next_index = last_index + 1
            except ValueError:
                print(f"⚠️  Last username '{last_username}' not found in current username list")
                print("🆕 Starting fresh analysis")
                return all_usernames, 0, {"resuming": False}
            
            # Get usernames that still need processing
            usernames_to_process = all_usernames[next_index:]
            already_processed = all_usernames[:next_index]
            
            if not usernames_to_process:
                print("✅ All usernames have already been processed!")
                return [], len(all_usernames), {"resuming": False, "completed": True}
            
            print(f"🔄 Resuming analysis from where we left off...")
            print(f"  Already processed: {len(already_processed)} users")
            print(f"  Remaining to process: {len(usernames_to_process)} users")
            print(f"  Resume point: {last_username} → {usernames_to_process[0]}")
            
            resume_info = {
                "resuming": True,
                "last_processed": last_username,
                "resume_index": next_index,
                "already_processed": len(already_processed),
                "remaining": len(usernames_to_process)
            }
            
            return usernames_to_process, len(already_processed), resume_info
            
        except Exception as e:
            print(f"❌ Error determining resume point: {str(e)}")
            print("🆕 Starting fresh analysis")
            return all_usernames, 0, {"resuming": False}

    def merge_with_previous_results(self, new_results, agpacket_dir="F:/DATA STORAGE/AGPacket"):
        """
        Merge new results with previous analysis results.
        
        Args:
            new_results (list): New analysis results
            agpacket_dir (str): Path to AGPacket directory
            
        Returns:
            list: Combined results
        """
        try:
            agpacket_path = Path(agpacket_dir)
            if not agpacket_path.exists():
                return new_results
            
            # Find the most recent analysis file
            analysis_files = list(agpacket_path.glob("reddit_user_analysis_*.json"))
            if not analysis_files:
                return new_results
            
            most_recent_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            print(f"🔄 Merging with previous results: {most_recent_file.name}")
            
            # Load previous results
            with open(most_recent_file, 'r', encoding='utf-8') as f:
                previous_results = json.load(f)
            
            if not previous_results:
                return new_results
            
            # Create a mapping of usernames to avoid duplicates
            username_map = {}
            
            # Add previous results
            for item in previous_results:
                if isinstance(item, dict) and 'username' in item:
                    username_map[item['username']] = item
            
            # Add new results (overwriting any duplicates)
            for item in new_results:
                if isinstance(item, dict) and 'username' in item:
                    username_map[item['username']] = item
            
            # Convert back to list
            combined_results = list(username_map.values())
            
            print(f"  Previous results: {len(previous_results)} users")
            print(f"  New results: {len(new_results)} users")
            print(f"  Combined results: {len(combined_results)} users")
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Error merging with previous results: {str(e)}")
            return new_results

    def save_progress_checkpoint(self, results, checkpoint_file, agpacket_dir="F:/DATA STORAGE/AGPacket"):
        """
        Save a progress checkpoint to allow resuming later.
        
        Args:
            results (list): Current analysis results
            checkpoint_file (str): Checkpoint file name
            agpacket_dir (str): Path to AGPacket directory
            
        Returns:
            str: Path to saved checkpoint file
        """
        try:
            checkpoint_path = Path(agpacket_dir) / checkpoint_file
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Progress checkpoint saved: {checkpoint_file}")
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {str(e)}")
            return None

    def analyze_users_with_resume(self, usernames, output_file=None, delay=1, checkpoint_interval=50):
        """
        Analyze a list of usernames with automatic progress saving and resume capability.
        
        Args:
            usernames (list): List of usernames to analyze
            output_file (str): Optional output file for results
            delay (float): Delay between API requests in seconds
            checkpoint_interval (int): Save checkpoint every N users
            
        Returns:
            list: List of user analysis results
        """
        if not self.access_token:
            print("❌ Not authenticated. Please call authenticate() first.")
            return []
        
        results = []
        total = len(usernames)
        
        print(f"\n🔍 Analyzing {total:,} usernames...")
        print("=" * 60)
        
        # Create checkpoint file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = f"reddit_user_analysis_checkpoint_{timestamp}.json"
        
        for i, username in enumerate(usernames, 1):
            print(f"[{i:,}/{total:,}] Analyzing: {username}")
            
            user_info = self.get_user_info(username)
            results.append(user_info)
            
            # Print summary
            if user_info['exists']:
                print(f"  ✅ Found: {user_info['total_posts']} posts, {user_info['total_comments']} comments")
            else:
                print(f"  ❌ {user_info['error']}")
            
            # Save checkpoint periodically
            if i % checkpoint_interval == 0:
                self.save_progress_checkpoint(results, checkpoint_file)
                print(f"  📊 Progress: {i/total*100:.1f}% complete ({i:,}/{total:,} users)")
            
            # Save progress to main output file
            if output_file:
                self._save_results(results, output_file)
            
            # Rate limiting delay
            if i < total:  # Don't delay after the last request
                time.sleep(delay)
        
        # Save final checkpoint
        self.save_progress_checkpoint(results, checkpoint_file)
        
        # Save final results
        if output_file:
            self._save_results(results, output_file)
            print(f"\n💾 Final results saved to: {output_file}")
        
        return results

    def cleanup_old_checkpoints(self, agpacket_dir="F:/DATA STORAGE/AGPacket", keep_recent=3):
        """
        Clean up old checkpoint files, keeping only the most recent ones.
        
        Args:
            agpacket_dir (str): Path to AGPacket directory
            keep_recent (int): Number of recent checkpoints to keep
        """
        try:
            agpacket_path = Path(agpacket_dir)
            if not agpacket_path.exists():
                return
            
            # Find all checkpoint files
            checkpoint_files = list(agpacket_path.glob("reddit_user_analysis_checkpoint_*.json"))
            
            if len(checkpoint_files) <= keep_recent:
                return
            
            # Sort by modification time and keep only the most recent
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            files_to_keep = checkpoint_files[:keep_recent]
            files_to_delete = checkpoint_files[keep_recent:]
            
            print(f"🧹 Cleaning up old checkpoint files...")
            print(f"  Keeping {len(files_to_keep)} recent checkpoints")
            print(f"  Deleting {len(files_to_delete)} old checkpoints")
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"    Deleted: {file_path.name}")
                except Exception as e:
                    print(f"    Failed to delete {file_path.name}: {str(e)}")
                    
        except Exception as e:
            print(f"❌ Error cleaning up checkpoints: {str(e)}")

    def print_resume_summary(self, resume_info, total_usernames, usernames_to_process):
        """
        Print a comprehensive summary of the resume operation.
        
        Args:
            resume_info (dict): Resume information
            total_usernames (int): Total usernames in the dataset
            usernames_to_process (list): Usernames to process
        """
        if not resume_info.get("resuming", False):
            return
        
        print(f"\n" + "=" * 60)
        print("🔄 RESUME OPERATION SUMMARY")
        print("=" * 60)
        
        already_processed = resume_info.get("already_processed", 0)
        remaining = resume_info.get("remaining", 0)
        last_processed = resume_info.get("last_processed", "Unknown")
        resume_index = resume_info.get("resume_index", 0)
        
        print(f"📊 Progress Statistics:")
        print(f"  Total usernames in dataset: {total_usernames:,}")
        print(f"  Already processed: {already_processed:,}")
        print(f"  Remaining to process: {remaining:,}")
        print(f"  Progress: {already_processed/total_usernames*100:.1f}% complete")
        
        print(f"\n📍 Resume Details:")
        print(f"  Last processed username: {last_processed}")
        print(f"  Resume index: {resume_index:,}")
        print(f"  Next username to process: {usernames_to_process[0] if usernames_to_process else 'None'}")
        
        if remaining > 0:
            estimated_time = remaining * 1.5 / 60  # Rough estimate: 1.5 seconds per user
            print(f"\n⏱️  Estimated completion time: {estimated_time:.1f} minutes")
            print(f"  Checkpoint frequency: Every 50 users")
            print(f"  Progress saved to: F:/DATA STORAGE/AGPacket/")
        
        print("=" * 60)

def main():
    """Main function to run the Reddit user analyzer."""
    
    print("🔍 Reddit User Analyzer")
    print("=" * 60)
    
    # Check for environment variables
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ Reddit API credentials not found in environment variables!")
        print("Please set the following environment variables:")
        print("  REDDIT_CLIENT_ID=your_client_id")
        print("  REDDIT_CLIENT_SECRET=your_client_secret")
        print("\nTo get these credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (select 'script')")
        print("3. Use the client ID and secret")
        return
    
    # Initialize analyzer
    analyzer = RedditUserAnalyzer(client_id, client_secret)
    
    # Authenticate
    if not analyzer.authenticate():
        return
    
    # Get usernames from CSV files
    usernames = analyzer.get_all_csv_usernames()
    
    if not usernames:
        print("No usernames found to analyze.")
        return
    
    # Check for resume capability
    print(f"\n🔍 Checking for previous analysis progress...")
    last_username, last_file, total_processed = analyzer.find_last_processed_username()
    
    # Determine which usernames to process
    usernames_to_process, already_processed, resume_info = analyzer.get_resume_usernames(usernames, last_username)
    
    if resume_info.get("completed", False):
        print("✅ Analysis already completed! All usernames have been processed.")
        return
    
    if not usernames_to_process:
        print("No usernames to process.")
        return
    
    # Show resume information
    if resume_info.get("resuming", False):
        print(f"\n🔄 RESUME MODE")
        print("=" * 60)
        print(f"Previous progress: {already_processed:,} users processed")
        print(f"Remaining work: {len(usernames_to_process):,} users to process")
        print(f"Resume point: {resume_info['last_processed']} → {usernames_to_process[0]}")
        print(f"Progress: {already_processed/(already_processed + len(usernames_to_process))*100:.1f}% complete")
        
        # Print detailed resume summary
        analyzer.print_resume_summary(resume_info, len(usernames), usernames_to_process)
    else:
        print(f"\n🆕 FRESH START")
        print("=" * 60)
        print(f"Total usernames to process: {len(usernames_to_process):,}")
    
    # Clean up old checkpoints
    analyzer.cleanup_old_checkpoints()
    
    # Ask user if they want to proceed
    response = input(f"\nDo you want to proceed with Reddit API analysis? (y/n): ").lower()
    
    if response != 'y':
        print("Analysis cancelled.")
        return
    
    # Set output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"F:/DATA STORAGE/AGPacket/reddit_user_analysis_{timestamp}.json"
    
    # Analyze users
    print(f"\nStarting analysis... (Results will be saved to {output_file})")
    
    if resume_info.get("resuming", False):
        print(f"🔄 Using resume-capable analysis with checkpoints...")
        results = analyzer.analyze_users_with_resume(usernames_to_process, output_file, delay=1, checkpoint_interval=50)
    else:
        print(f"🆕 Using standard analysis...")
        results = analyzer.analyze_users(usernames_to_process, output_file, delay=1)
    
    # Merge with previous results if resuming
    if resume_info.get("resuming", False):
        print(f"\n🔄 Merging new results with previous analysis...")
        combined_results = analyzer.merge_with_previous_results(results)
        
        # Save the combined results to a new file
        combined_output_file = f"F:/DATA STORAGE/AGPacket/reddit_user_analysis_combined_{timestamp}.json"
        analyzer._save_results(combined_results, combined_output_file)
        print(f"💾 Combined results saved to: {combined_output_file}")
        
        # Use combined results for content extraction
        results = combined_results
        output_file = combined_output_file
    else:
        print(f"💾 New results saved to: {output_file}")
    
    # Generate summary report
    analyzer.generate_summary_report(results)
    
    # Extract content samples
    print(f"\n" + "=" * 60)
    print("📝 CONTENT EXTRACTION")
    print("=" * 60)
    
    # Find existing database files for duplicate checking
    existing_db_files = []
    data_dir = Path("F:/DATA STORAGE/AGPacket")
    
    # Look for existing content sample files
    for file_path in data_dir.glob("reddit_content_samples_*.json"):
        existing_db_files.append(file_path)
    
    # Look for existing user analysis files that might contain content
    for file_path in data_dir.glob("reddit_user_analysis_*.json"):
        if file_path.name != Path(output_file).name:  # Don't check against current output
            existing_db_files.append(file_path)
    
    if existing_db_files:
        # Use the most recent file for duplicate checking
        most_recent_db = max(existing_db_files, key=lambda x: x.stat().st_mtime)
        print(f"🔍 Found existing database for duplicate checking: {most_recent_db.name}")
        print(f"   Last modified: {datetime.fromtimestamp(most_recent_db.stat().st_mtime)}")
    else:
        most_recent_db = None
        print("⚙️  No existing database files found. Skipping duplicate check.")
    
    content_samples = analyzer.extract_user_content_samples(
        results, 
        min_words=5, 
        max_words=500, 
        max_samples_per_user=5, 
        existing_database_file=str(most_recent_db) if most_recent_db else None
    )
    
    if content_samples:
        # Print content summary
        analyzer.print_content_summary(content_samples)
        
        # Save content samples
        content_output_file = f"F:/DATA STORAGE/AGPacket/reddit_content_samples_{timestamp}.json"
        analyzer.save_content_samples(content_samples, content_output_file)
        
        print(f"\n✅ Content extraction complete!")
        print(f"📁 User analysis saved to: {output_file}")
        print(f"📝 Content samples saved to: {content_output_file}")
    else:
        print("❌ No content samples found.")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
