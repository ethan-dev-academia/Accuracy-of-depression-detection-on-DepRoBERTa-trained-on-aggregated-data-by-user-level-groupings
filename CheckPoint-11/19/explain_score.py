"""Explain what 'score' means in the Reddit data."""
import json
from pathlib import Path
from collections import Counter

output_dir = Path(r"F:\DATA STORAGE\AGG_PACKET\labeling_outputs")

# Load a sample from all_labeled_users to see the raw structure
with open(output_dir / "all_labeled_users.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("WHAT IS 'SCORE' IN REDDIT DATA?")
print("="*80)

print("""
In Reddit data, 'score' refers to the upvote/downvote score of a post or comment.

HOW IT WORKS:
- Users can upvote or downvote posts and comments
- Score = Total upvotes - Total downvotes
- Higher score = more popular/well-received content
- Score can be positive, negative, or zero
- Score of 0 usually means no votes or equal upvotes/downvotes

EXAMPLE:
- Score of 100 = 100 more upvotes than downvotes
- Score of -5 = 5 more downvotes than upvotes
- Score of 0 = Equal votes or no votes
""")

# Find a record with posts/comments
sample_rec = None
for rec in data[:100]:
    if rec.get('posts') or rec.get('comments'):
        sample_rec = rec
        break

if sample_rec:
    print("\n" + "="*80)
    print("EXAMPLE FROM YOUR DATA")
    print("="*80)
    
    print(f"\nUser: {sample_rec.get('username', 'unknown')}")
    
    # Show post scores
    if sample_rec.get('posts'):
        print(f"\nPosts ({len(sample_rec['posts'])} total):")
        scores = []
        for i, post in enumerate(sample_rec['posts'][:5], 1):
            score = post.get('score', 0)
            scores.append(score)
            title = post.get('title', '')[:60]
            print(f"  Post {i}:")
            print(f"    Title: {title}...")
            print(f"    Score: {score}")
            print(f"    Meaning: {score} upvotes more than downvotes")
        
        print(f"\nScore distribution (first 5 posts):")
        print(f"  Min: {min(scores)}")
        print(f"  Max: {max(scores)}")
        print(f"  Average: {sum(scores)/len(scores):.1f}")
    
    # Show comment scores
    if sample_rec.get('comments'):
        print(f"\nComments ({len(sample_rec['comments'])} total):")
        scores = []
        for i, comment in enumerate(sample_rec['comments'][:5], 1):
            score = comment.get('score', 0)
            scores.append(score)
            body = (comment.get('content') or comment.get('body', ''))[:60]
            print(f"  Comment {i}:")
            print(f"    Text: {body}...")
            print(f"    Score: {score}")
            print(f"    Meaning: {score} upvotes more than downvotes")
        
        if scores:
            print(f"\nScore distribution (first 5 comments):")
            print(f"  Min: {min(scores)}")
            print(f"  Max: {max(scores)}")
            print(f"  Average: {sum(scores)/len(scores):.1f}")

# Check how score is used in aggregation
print("\n" + "="*80)
print("HOW SCORE IS USED IN YOUR TRAINING DATA")
print("="*80)

print("""
In the aggregation process (prepare_training_dataset.py), score is used to:
1. Sort posts/comments by popularity (higher score = more popular)
2. Prioritize more engaging content when aggregating user text
3. Help filter out low-quality or controversial content

However, in the final training dataset (train.json, val.json, test.json), 
the score field is NOT included - only the aggregated text and label are kept.

The score was used during processing but is not part of the final training examples.
""")

# Show structure comparison
print("\n" + "="*80)
print("DATA STRUCTURE COMPARISON")
print("="*80)

print("\n1. RAW DATA (all_labeled_users.json):")
print("   Each post/comment has a 'score' field")
print("   Example post structure:")
print("   {")
print("     'title': '...',")
print("     'content': '...',")
print("     'score': 42,  <-- Reddit upvote/downvote score")
print("     'subreddit': '...',")
print("     'created_utc': ...")
print("   }")

print("\n2. TRAINING DATA (train.json, val.json, test.json):")
print("   Score is NOT included - only aggregated text")
print("   Example structure:")
print("   {")
print("     'text': '...',  <-- All posts/comments combined")
print("     'label': 0 or 1,")
print("     'user_id': '...',")
print("     'segments': [...]")
print("   }")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Score = Reddit upvote/downvote count (popularity metric)
- Used during data processing to prioritize popular content
- NOT included in final training dataset
- Only text content and labels are used for training
""")

