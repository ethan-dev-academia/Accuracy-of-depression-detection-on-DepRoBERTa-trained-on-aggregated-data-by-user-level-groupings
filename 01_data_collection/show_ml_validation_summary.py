#!/usr/bin/env python3
"""
Quick summary of ML validation results
"""

import json
from pathlib import Path

def show_summary():
    """Show a summary of the ML validation results."""
    
    # Find the most recent validation report
    reports = list(Path('.').glob('ml_validation_report_*.json'))
    if not reports:
        print("❌ No validation reports found!")
        return
    
    latest_report = max(reports, key=lambda x: x.stat().st_mtime)
    print(f"📋 Loading latest validation report: {latest_report.name}")
    
    try:
        with open(latest_report, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f"❌ Error loading report: {e}")
        return
    
    print("\n" + "="*80)
    print("🤖 REDDIT DATA ML TRAINING VALIDATION SUMMARY")
    print("="*80)
    
    # Summary statistics
    summary = report['summary_statistics']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   • Total files analyzed: {report['total_files_analyzed']}")
    print(f"   • Valid files: {summary.get('total_valid_files', 0)}")
    print(f"   • Corrupted files: {summary.get('total_corrupted_files', 0)}")
    print(f"   • Total data size: {summary.get('total_data_size_mb', 0):.1f} MB")
    print(f"   • Average ML readiness score: {summary.get('average_ml_score', 0):.1f}/100")
    
    # Top quality files
    print(f"\n🏆 TOP QUALITY FILES FOR ML TRAINING:")
    ml_scores = []
    for file_name, ml_assessment in report['ml_readiness_assessment'].items():
        ml_scores.append((file_name, ml_assessment['overall_score']))
    
    ml_scores.sort(key=lambda x: x[1], reverse=True)
    for i, (file_name, score) in enumerate(ml_scores[:5], 1):
        file_size = next((f['file_size_mb'] for f in report['file_validation_results'] 
                         if f['file_name'] == file_name), 0)
        print(f"   {i}. {file_name}")
        print(f"      Score: {score:.1f}/100 | Size: {file_size:.1f} MB")
    
    # Content quality insights
    print(f"\n📈 CONTENT QUALITY INSIGHTS:")
    if report['content_quality_analysis']:
        best_file = ml_scores[0][0]
        quality_data = report['content_quality_analysis'][best_file]
        
        print(f"   • Total users: {quality_data.get('total_users', 0):,}")
        print(f"   • Active users: {quality_data.get('active_users', 0):,}")
        print(f"   • Subreddit diversity: {len(quality_data.get('subreddit_diversity', []))}")
        
        # Content length stats
        post_stats = quality_data.get('content_length_stats', {}).get('posts_stats', {})
        if post_stats:
            print(f"   • Average post length: {post_stats.get('mean', 0):.1f} characters")
            print(f"   • Post count: {post_stats.get('count', 0):,}")
        
        # Temporal coverage
        temporal = quality_data.get('temporal_coverage', {})
        if temporal:
            print(f"   • Time span: {len(temporal)} months")
            print(f"   • Date range: {min(temporal.keys())} to {max(temporal.keys())}")
    
    # ML readiness breakdown
    print(f"\n🎯 ML READINESS BREAKDOWN:")
    if report['ml_readiness_assessment']:
        best_assessment = report['ml_readiness_assessment'][ml_scores[0][0]]
        print(f"   • Data quantity: {best_assessment['data_quantity']:.0f}/25")
        print(f"   • Data quality: {best_assessment['data_quality']:.0f}/25")
        print(f"   • Content diversity: {best_assessment['content_diversity']:.0f}/25")
        print(f"   • Temporal coverage: {best_assessment['temporal_coverage']:.0f}/25")
    
    # Recommendations
    print(f"\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"   {i}. {rec}")
    
    # ML use cases
    print(f"\n🚀 RECOMMENDED ML USE CASES:")
    if report['ml_readiness_assessment']:
        best_file = ml_scores[0][0]
        use_cases = report['ml_readiness_assessment'][best_file]['ml_use_cases']
        for i, use_case in enumerate(use_cases[:6], 1):
            print(f"   {i}. {use_case}")
    
    print("\n" + "="*80)
    print(f"📁 Full report: {latest_report.name}")
    print("="*80)

if __name__ == "__main__":
    show_summary()
