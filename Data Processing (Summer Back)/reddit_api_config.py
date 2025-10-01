#!/usr/bin/env python3
"""
Configuration file for Reddit API keys used in parallel processing.
You can add more API keys here for additional instances.
"""

# Reddit API Configuration
REDDIT_API_KEYS = {
    1: {
        'client_id': 'foI3sdH3CG5V-JQZN7ymeg',
        'client_secret': 'dspoNLvUGgyOdsFmZTMaP2gM-klwjA',
        'user_agent': 'RedditUserAnalyzer/1.0-Instance1'
    },
    2: {
        'client_id': 'foI3sdH3CG5V-JQZN7ymeg',  # Replace with actual second API key
        'client_secret': 'dspoNLvUGgyOdsFmZTMaP2gM-klwjA',  # Replace with actual second secret
        'user_agent': 'RedditUserAnalyzer/1.0-Instance2'
    }
    # Add more instances as needed:
    # 3: {
    #     'client_id': 'your_third_client_id',
    #     'client_secret': 'your_third_client_secret',
    #     'user_agent': 'RedditUserAnalyzer/1.0-Instance3'
    # }
}

def get_api_key(instance_id):
    """Get API key configuration for a specific instance."""
    return REDDIT_API_KEYS.get(instance_id)

def get_all_instance_ids():
    """Get list of all available instance IDs."""
    return list(REDDIT_API_KEYS.keys())

def validate_config():
    """Validate that all API keys are properly configured."""
    for instance_id, config in REDDIT_API_KEYS.items():
        if not config.get('client_id') or not config.get('client_secret'):
            print(f"⚠️  Warning: Instance {instance_id} has incomplete configuration")
            return False
    return True

if __name__ == "__main__":
    print("🔑 Reddit API Configuration")
    print("=" * 40)
    
    for instance_id, config in REDDIT_API_KEYS.items():
        print(f"Instance {instance_id}:")
        print(f"  Client ID: {config['client_id'][:10]}...")
        print(f"  User Agent: {config['user_agent']}")
        print()
    
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has issues")
