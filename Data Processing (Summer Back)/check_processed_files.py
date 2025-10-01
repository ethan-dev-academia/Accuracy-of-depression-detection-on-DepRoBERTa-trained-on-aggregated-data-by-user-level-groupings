#!/usr/bin/env python3
"""
Reddit Data ML Training Validation Script

This script validates Reddit user analysis data for machine learning training.
It performs comprehensive checks on data quality, content analysis, and ML readiness.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import re
import statistics
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

class RedditDataValidator:
    """Validates Reddit data for ML training readiness."""
    
    def __init__(self, data_dir: str = "F:/DATA STORAGE/AGPacket"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        self.data_stats = {}
        self.ml_readiness_score = 0.0
        
    def scan_data_files(self) -> List[Path]:
        """Scan for Reddit analysis data files."""
        print("🔍 Scanning for Reddit data files...")
        
        data_files = []
        for file_path in self.data_dir.glob("reddit_user_analysis_*.json"):
            if not file_path.name.endswith('.backup'):
                data_files.append(file_path)
        
        print(f"📁 Found {len(data_files)} data files")
        return sorted(data_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def validate_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """Validate individual file integrity."""
        print(f"🔍 Validating: {file_path.name}")
        
        result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            'is_valid_json': False,
            'parse_errors': [],
            'data_structure': {},
            'validation_status': 'unknown'
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result['is_valid_json'] = True
            result['data_structure'] = self._analyze_data_structure(data)
            result['validation_status'] = 'valid'
            
        except json.JSONDecodeError as e:
            result['parse_errors'].append(f"JSON decode error: {str(e)}")
            result['validation_status'] = 'corrupted'
        except Exception as e:
            result['parse_errors'].append(f"Unexpected error: {str(e)}")
            result['validation_status'] = 'error'
        
        return result
    
    def _analyze_data_structure(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze the structure of the data."""
        if not isinstance(data, list):
            return {'error': 'Data is not a list'}
        
        structure = {
            'total_users': len(data),
            'user_fields': set(),
            'post_fields': set(),
            'comment_fields': set(),
            'field_coverage': {},
            'data_types': {}
        }
        
        if not data:
            return structure
        
        # Analyze first user to get field structure
        first_user = data[0]
        structure['user_fields'] = set(first_user.keys())
        
        # Analyze posts and comments structure
        if 'posts' in first_user and first_user['posts']:
            structure['post_fields'] = set(first_user['posts'][0].keys())
        
        if 'comments' in first_user and first_user['comments']:
            structure['comment_fields'] = set(first_user['comments'][0].keys())
        
        # Analyze field coverage across all users
        for field in structure['user_fields']:
            coverage = sum(1 for user in data if field in user and user[field] is not None)
            structure['field_coverage'][field] = {
                'present': coverage,
                'missing': len(data) - coverage,
                'coverage_pct': (coverage / len(data)) * 100
            }
        
        return structure
    
    def analyze_content_quality(self, file_path: Path) -> Dict[str, Any]:
        """Analyze content quality for ML training."""
        print(f"📊 Analyzing content quality: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return {'error': 'Could not load file'}
        
        quality_metrics = {
            'total_users': len(data),
            'active_users': 0,
            'content_distribution': {},
            'text_quality': {},
            'subreddit_diversity': set(),
            'temporal_coverage': {},
            'engagement_metrics': {'scores': []},
            'content_length_stats': {'posts': [], 'comments': []},
            'language_quality': {}
        }
        
        # Analyze each user
        for user in data:
            if not isinstance(user, dict):
                continue
                
            # Check if user is active
            posts_count = len(user.get('posts', []))
            comments_count = len(user.get('comments', []))
            if posts_count > 0 or comments_count > 0:
                quality_metrics['active_users'] += 1
            
            # Analyze posts
            for post in user.get('posts', []):
                if isinstance(post, dict):
                    self._analyze_content_item(post, quality_metrics, 'post')
            
            # Analyze comments
            for comment in user.get('comments', []):
                if isinstance(comment, dict):
                    self._analyze_content_item(comment, quality_metrics, 'comment')
        
        # Convert sets to lists for JSON serialization
        quality_metrics['subreddit_diversity'] = list(quality_metrics['subreddit_diversity'])
        
        # Calculate statistics
        self._calculate_content_statistics(quality_metrics)
        
        return quality_metrics
    
    def _analyze_content_item(self, item: Dict, metrics: Dict, item_type: str):
        """Analyze individual content item."""
        # Subreddit diversity
        if 'subreddit' in item:
            metrics['subreddit_diversity'].add(item['subreddit'])
        
        # Content length
        content = item.get('content', '') or item.get('body', '')
        if content and content != '[removed]' and content != '[deleted]':
            content_length = len(content.strip())
            if 'content_length_stats' not in metrics:
                metrics['content_length_stats'] = {'posts': [], 'comments': []}
            metrics['content_length_stats'][f'{item_type}s'].append(content_length)
        
        # Temporal coverage
        if 'created_utc' in item:
            try:
                timestamp = float(item['created_utc'])
                date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                year_month = f"{date.year}-{date.month:02d}"
                if 'temporal_coverage' not in metrics:
                    metrics['temporal_coverage'] = {}
                metrics['temporal_coverage'][year_month] = metrics['temporal_coverage'].get(year_month, 0) + 1
            except:
                pass
        
        # Engagement metrics
        if 'score' in item:
            score = item['score']
            if isinstance(score, (int, float)):
                if 'engagement_metrics' not in metrics:
                    metrics['engagement_metrics'] = {'scores': []}
                metrics['engagement_metrics']['scores'].append(score)
    
    def _calculate_content_statistics(self, metrics: Dict):
        """Calculate statistical measures for content quality."""
        # Content length statistics
        for content_type in ['posts', 'comments']:
            lengths = metrics['content_length_stats'].get(content_type, [])
            if lengths:
                metrics['content_length_stats'][f'{content_type}_stats'] = {
                    'count': len(lengths),
                    'mean': statistics.mean(lengths),
                    'median': statistics.median(lengths),
                    'min': min(lengths),
                    'max': max(lengths),
                    'std': statistics.stdev(lengths) if len(lengths) > 1 else 0
                }
        
        # Engagement statistics
        scores = metrics['engagement_metrics'].get('scores', [])
        if scores:
            metrics['engagement_metrics']['score_stats'] = {
                'count': len(scores),
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0
            }
    
    def assess_ml_readiness(self, quality_metrics: Dict) -> Dict[str, Any]:
        """Assess if the data is ready for ML training."""
        print("🤖 Assessing ML training readiness...")
        
        ml_assessment = {
            'overall_score': 0.0,
            'data_quantity': 0.0,
            'data_quality': 0.0,
            'content_diversity': 0.0,
            'temporal_coverage': 0.0,
            'recommendations': [],
            'ml_use_cases': [],
            'data_limitations': []
        }
        
        # Data quantity score (0-25 points)
        total_users = quality_metrics.get('total_users', 0)
        active_users = quality_metrics.get('active_users', 0)
        
        if total_users >= 10000:
            ml_assessment['data_quantity'] = 25.0
        elif total_users >= 5000:
            ml_assessment['data_quantity'] = 20.0
        elif total_users >= 1000:
            ml_assessment['data_quantity'] = 15.0
        elif total_users >= 500:
            ml_assessment['data_quantity'] = 10.0
        else:
            ml_assessment['data_quantity'] = 5.0
        
        # Data quality score (0-25 points)
        content_length_stats = quality_metrics.get('content_length_stats', {})
        post_stats = content_length_stats.get('posts_stats', {})
        
        if post_stats.get('count', 0) > 0:
            avg_length = post_stats.get('mean', 0)
            if avg_length >= 100:
                ml_assessment['data_quality'] = 25.0
            elif avg_length >= 50:
                ml_assessment['data_quality'] = 20.0
            elif avg_length >= 20:
                ml_assessment['data_quality'] = 15.0
            else:
                ml_assessment['data_quality'] = 10.0
        else:
            ml_assessment['data_quality'] = 5.0
        
        # Content diversity score (0-25 points)
        subreddit_count = len(quality_metrics.get('subreddit_diversity', []))
        if subreddit_count >= 100:
            ml_assessment['content_diversity'] = 25.0
        elif subreddit_count >= 50:
            ml_assessment['content_diversity'] = 20.0
        elif subreddit_count >= 20:
            ml_assessment['content_diversity'] = 15.0
        elif subreddit_count >= 10:
            ml_assessment['content_diversity'] = 10.0
        else:
            ml_assessment['content_diversity'] = 5.0
        
        # Temporal coverage score (0-25 points)
        temporal_coverage = quality_metrics.get('temporal_coverage', {})
        if len(temporal_coverage) >= 24:  # 2 years of monthly data
            ml_assessment['temporal_coverage'] = 25.0
        elif len(temporal_coverage) >= 12:  # 1 year of monthly data
            ml_assessment['temporal_coverage'] = 20.0
        elif len(temporal_coverage) >= 6:  # 6 months of monthly data
            ml_assessment['temporal_coverage'] = 15.0
        elif len(temporal_coverage) >= 3:  # 3 months of monthly data
            ml_assessment['temporal_coverage'] = 10.0
        else:
            ml_assessment['temporal_coverage'] = 5.0
        
        # Calculate overall score
        ml_assessment['overall_score'] = (
            ml_assessment['data_quantity'] +
            ml_assessment['data_quality'] +
            ml_assessment['content_diversity'] +
            ml_assessment['temporal_coverage']
        )
        
        # Generate recommendations
        self._generate_ml_recommendations(ml_assessment, quality_metrics)
        
        return ml_assessment
    
    def _generate_ml_recommendations(self, assessment: Dict, quality_metrics: Dict):
        """Generate ML training recommendations."""
        recommendations = []
        ml_use_cases = []
        limitations = []
        
        # Data quantity recommendations
        if assessment['data_quantity'] < 20:
            recommendations.append("Consider collecting more user data for robust ML training")
            limitations.append("Limited training data may lead to overfitting")
        
        # Data quality recommendations
        if assessment['data_quality'] < 20:
            recommendations.append("Content length is limited - consider filtering for longer posts")
            limitations.append("Short content may not provide sufficient context for ML models")
        
        # Diversity recommendations
        if assessment['content_diversity'] < 20:
            recommendations.append("Limited subreddit diversity - consider expanding data collection")
            limitations.append("Low diversity may bias models toward specific topics")
        
        # ML use cases based on data characteristics
        if assessment['overall_score'] >= 80:
            ml_use_cases.extend([
                "User behavior prediction",
                "Content recommendation systems",
                "Sentiment analysis",
                "Topic modeling",
                "User clustering and segmentation"
            ])
        elif assessment['overall_score'] >= 60:
            ml_use_cases.extend([
                "Basic sentiment analysis",
                "Simple topic classification",
                "User activity prediction"
            ])
        else:
            ml_use_cases.append("Limited ML applications - data quality needs improvement")
        
        assessment['recommendations'] = recommendations
        assessment['ml_use_cases'] = ml_use_cases
        assessment['data_limitations'] = limitations
    
    def generate_validation_report(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("📋 Generating validation report...")
        
        report = {
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_files_analyzed': len(file_paths),
            'file_validation_results': [],
            'content_quality_analysis': {},
            'ml_readiness_assessment': {},
            'summary_statistics': {},
            'recommendations': []
        }
        
        # Validate each file
        for file_path in file_paths:
            file_result = self.validate_file_integrity(file_path)
            report['file_validation_results'].append(file_result)
            
            # Analyze content quality for valid files
            if file_result['validation_status'] == 'valid':
                quality_metrics = self.analyze_content_quality(file_path)
                report['content_quality_analysis'][file_path.name] = quality_metrics
                
                # Assess ML readiness
                ml_assessment = self.assess_ml_readiness(quality_metrics)
                report['ml_readiness_assessment'][file_path.name] = ml_assessment
        
        # Generate summary statistics
        report['summary_statistics'] = self._generate_summary_statistics(report)
        
        # Generate overall recommendations
        report['recommendations'] = self._generate_overall_recommendations(report)
        
        return report
    
    def _generate_summary_statistics(self, report: Dict) -> Dict[str, Any]:
        """Generate summary statistics across all files."""
        valid_files = [f for f in report['file_validation_results'] if f['validation_status'] == 'valid']
        
        if not valid_files:
            return {'error': 'No valid files found'}
        
        summary = {
            'total_valid_files': len(valid_files),
            'total_corrupted_files': len([f for f in report['file_validation_results'] if f['validation_status'] == 'corrupted']),
            'total_data_size_mb': sum(f['file_size_mb'] for f in valid_files),
            'average_file_size_mb': statistics.mean(f['file_size_mb'] for f in valid_files),
            'ml_readiness_scores': []
        }
        
        # Collect ML readiness scores
        for file_name, ml_assessment in report['ml_readiness_assessment'].items():
            summary['ml_readiness_scores'].append(ml_assessment['overall_score'])
        
        if summary['ml_readiness_scores']:
            summary['average_ml_score'] = statistics.mean(summary['ml_readiness_scores'])
            summary['best_ml_score'] = max(summary['ml_readiness_scores'])
            summary['worst_ml_score'] = min(summary['ml_readiness_scores'])
        
        return summary
    
    def _generate_overall_recommendations(self, report: Dict) -> List[str]:
        """Generate overall recommendations based on all data."""
        recommendations = []
        
        summary = report['summary_statistics']
        
        if summary.get('total_corrupted_files', 0) > 0:
            recommendations.append(f"Fix {summary['total_corrupted_files']} corrupted files before ML training")
        
        if summary.get('average_ml_score', 0) < 60:
            recommendations.append("Overall data quality is below optimal for ML training")
        
        if summary.get('total_data_size_mb', 0) < 100:
            recommendations.append("Consider collecting more data for comprehensive ML training")
        
        recommendations.append("Use the best quality files (highest ML readiness scores) for initial ML experiments")
        recommendations.append("Implement data preprocessing pipeline to clean and standardize content")
        
        return recommendations
    
    def save_validation_report(self, report: Dict, output_file: str = None) -> str:
        """Save validation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ml_validation_report_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Validation report saved: {output_path}")
        return str(output_path)
    
    def print_summary(self, report: Dict):
        """Print a human-readable summary of the validation results."""
        print("\n" + "="*80)
        print("🤖 REDDIT DATA ML TRAINING VALIDATION REPORT")
        print("="*80)
        
        summary = report['summary_statistics']
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total files analyzed: {report['total_files_analyzed']}")
        print(f"   Valid files: {summary.get('total_valid_files', 0)}")
        print(f"   Corrupted files: {summary.get('total_corrupted_files', 0)}")
        print(f"   Total data size: {summary.get('total_data_size_mb', 0):.1f} MB")
        print(f"   Average ML readiness score: {summary.get('average_ml_score', 0):.1f}/100")
        
        print(f"\n🏆 BEST QUALITY FILES:")
        ml_scores = []
        for file_name, ml_assessment in report['ml_readiness_assessment'].items():
            ml_scores.append((file_name, ml_assessment['overall_score']))
        
        ml_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (file_name, score) in enumerate(ml_scores[:3]):
            print(f"   {i+1}. {file_name}: {score:.1f}/100")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🚀 ML USE CASES:")
        if report['ml_readiness_assessment']:
            # Get use cases from the best file
            best_file = ml_scores[0][0]
            use_cases = report['ml_readiness_assessment'][best_file]['ml_use_cases']
            for i, use_case in enumerate(use_cases[:5], 1):
                print(f"   {i}. {use_case}")
        
        print("="*80)

def main():
    """Main function to run the validation."""
    print("🔍 Reddit Data ML Training Validator")
    print("="*50)
    
    # Initialize validator
    validator = RedditDataValidator()
    
    # Scan for data files
    data_files = validator.scan_data_files()
    
    if not data_files:
        print("❌ No Reddit data files found!")
        return
    
    # Generate validation report
    report = validator.generate_validation_report(data_files)
    
    # Save report
    output_file = validator.save_validation_report(report)
    
    # Print summary
    validator.print_summary(report)
    
    print(f"\n✅ Validation complete! Full report saved to: {output_file}")

if __name__ == "__main__":
    main()
