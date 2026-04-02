#!/usr/bin/env python3
"""
Analyze and compare results from Fast Weights experiments
"""

import json
import os
import argparse
from pathlib import Path
import re
from collections import defaultdict


def parse_log_file(log_path):
    """Parse a log file and extract metrics"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    metrics = {
        'name': log_path.stem,
        'num_params': None,
        'train_bpc': [],
        'valid_bpc': [],
        'test_bpc': None,
        'best_valid_bpc': None
    }
    
    # Extract number of parameters
    match = re.search(r'Number of parameters:\s*([\d,]+)', content)
    if match:
        metrics['num_params'] = match.group(1)
    
    # Extract BPC values per epoch
    for line in content.split('\n'):
        if 'Train BPC:' in line:
            match = re.search(r'Train BPC:\s*([\d.]+)', line)
            if match:
                metrics['train_bpc'].append(float(match.group(1)))
        
        if 'Valid BPC:' in line:
            match = re.search(r'Valid BPC:\s*([\d.]+)', line)
            if match:
                metrics['valid_bpc'].append(float(match.group(1)))
        
        if 'Saved best model with valid BPC:' in line:
            match = re.search(r'valid BPC:\s*([\d.]+)', line)
            if match:
                metrics['best_valid_bpc'] = float(match.group(1))
        
        if 'Test BPC:' in line and 'Final Test' in content.split(line)[0][-200:]:
            match = re.search(r'Test BPC:\s*([\d.]+)', line)
            if match:
                metrics['test_bpc'] = float(match.group(1))
    
    # Get best valid BPC if not explicitly saved
    if metrics['best_valid_bpc'] is None and metrics['valid_bpc']:
        metrics['best_valid_bpc'] = min(metrics['valid_bpc'])
    
    # Get final train BPC
    if metrics['train_bpc']:
        metrics['final_train_bpc'] = metrics['train_bpc'][-1]
    else:
        metrics['final_train_bpc'] = None
    
    # Get final valid BPC
    if metrics['valid_bpc']:
        metrics['final_valid_bpc'] = metrics['valid_bpc'][-1]
    else:
        metrics['final_valid_bpc'] = None
    
    return metrics


def analyze_experiment_logs(log_dir):
    """Analyze all log files in a directory"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist")
        return
    
    # Find all .log files
    log_files = list(log_dir.glob('*.log'))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    print(f"\nAnalyzing {len(log_files)} log files from {log_dir}\n")
    
    # Parse all logs
    results = []
    for log_file in sorted(log_files):
        metrics = parse_log_file(log_file)
        results.append(metrics)
    
    # Group by model type
    rnn_results = [r for r in results if 'rnn' in r['name'].lower() and 'lstm' not in r['name'].lower()]
    lstm_results = [r for r in results if 'lstm' in r['name'].lower()]
    
    # Print results
    print("="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    
    if rnn_results:
        print("\n### RNN MODELS ###\n")
        print(f"{'Model':<30} {'Params':<15} {'Best Valid BPC':<20} {'Test BPC':<15}")
        print("-"*100)
        for r in rnn_results:
            model_name = r['name']
            params = r['num_params'] or 'N/A'
            best_valid = f"{r['best_valid_bpc']:.4f}" if r['best_valid_bpc'] else 'N/A'
            test = f"{r['test_bpc']:.4f}" if r['test_bpc'] else 'N/A'
            print(f"{model_name:<30} {params:<15} {best_valid:<20} {test:<15}")
    
    if lstm_results:
        print("\n### LSTM MODELS ###\n")
        print(f"{'Model':<30} {'Params':<15} {'Best Valid BPC':<20} {'Test BPC':<15}")
        print("-"*100)
        for r in lstm_results:
            model_name = r['name']
            params = r['num_params'] or 'N/A'
            best_valid = f"{r['best_valid_bpc']:.4f}" if r['best_valid_bpc'] else 'N/A'
            test = f"{r['test_bpc']:.4f}" if r['test_bpc'] else 'N/A'
            print(f"{model_name:<30} {params:<15} {best_valid:<20} {test:<15}")
    
    # Print comparison
    print("\n" + "="*100)
    print("COMPARISON")
    print("="*100)
    
    # Find best models
    valid_results = [r for r in results if r['best_valid_bpc'] is not None]
    if valid_results:
        best_model = min(valid_results, key=lambda x: x['best_valid_bpc'])
        print(f"\n🏆 Best Model (by Valid BPC): {best_model['name']}")
        print(f"   Valid BPC: {best_model['best_valid_bpc']:.4f}")
        if best_model['test_bpc']:
            print(f"   Test BPC: {best_model['test_bpc']:.4f}")
    
    # Compare Standard vs Fast Weights
    print("\n### Fast Weights Improvement ###\n")
    
    # RNN comparison
    standard_rnn = next((r for r in rnn_results if 'standard_rnn' in r['name']), None)
    fast_rnns = [r for r in rnn_results if 'fast_rnn' in r['name']]
    
    if standard_rnn and fast_rnns:
        print("RNN Models:")
        baseline_bpc = standard_rnn['best_valid_bpc']
        if baseline_bpc:
            print(f"  Baseline (Standard RNN): {baseline_bpc:.4f} BPC")
            for fast_rnn in fast_rnns:
                if fast_rnn['best_valid_bpc']:
                    improvement = baseline_bpc - fast_rnn['best_valid_bpc']
                    improvement_pct = (improvement / baseline_bpc) * 100
                    print(f"  {fast_rnn['name']}: {fast_rnn['best_valid_bpc']:.4f} BPC "
                          f"(Δ {improvement:+.4f}, {improvement_pct:+.2f}%)")
    
    # LSTM comparison
    standard_lstm = next((r for r in lstm_results if r['name'].startswith('lstm_') and 'fast' not in r['name']), None)
    fast_lstms = [r for r in lstm_results if 'fast_lstm' in r['name']]
    
    if standard_lstm and fast_lstms:
        print("\nLSTM Models:")
        baseline_bpc = standard_lstm['best_valid_bpc']
        if baseline_bpc:
            print(f"  Baseline (Standard LSTM): {baseline_bpc:.4f} BPC")
            for fast_lstm in fast_lstms:
                if fast_lstm['best_valid_bpc']:
                    improvement = baseline_bpc - fast_lstm['best_valid_bpc']
                    improvement_pct = (improvement / baseline_bpc) * 100
                    print(f"  {fast_lstm['name']}: {fast_lstm['best_valid_bpc']:.4f} BPC "
                          f"(Δ {improvement:+.4f}, {improvement_pct:+.2f}%)")
    
    print("\n" + "="*100)
    
    # Save summary to JSON
    summary_file = log_dir / 'summary.json'
    summary = {
        'log_directory': str(log_dir),
        'num_experiments': len(results),
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*100)


def find_latest_log_dir(base_dir='experiment_logs'):
    """Find the most recent experiment log directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Sort by name (timestamp-based) and return latest
    return sorted(subdirs)[-1]


def main():
    parser = argparse.ArgumentParser(description='Analyze Fast Weights experiment results')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Path to experiment log directory (default: latest in experiment_logs/)')
    
    args = parser.parse_args()
    
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            print("Error: No experiment logs found in experiment_logs/")
            print("Please run experiments first or specify --log_dir")
            return
        print(f"Using latest log directory: {log_dir}")
    
    analyze_experiment_logs(log_dir)


if __name__ == '__main__':
    main()
