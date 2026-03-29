#!/usr/bin/env python3
"""
Analyze and visualize hyperparameter tuning results
"""

import json
import os
import sys
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all trial results from directory"""
    all_results = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith('trial_') and filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_results.append(json.load(f))
    return all_results

def plot_training_curves(results: List[Dict[str, Any]], output_dir: str):
    """Plot training and validation curves for all trials"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training curves
    plt.subplot(1, 3, 1)
    for result in results:
        epochs = range(1, len(result['history']['train_loss']) + 1)
        plt.plot(epochs, result['history']['train_loss'], 
                alpha=0.6, label=result['config']['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Training BPC')
    plt.title('Training Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation curves
    plt.subplot(1, 3, 2)
    for result in results:
        epochs = range(1, len(result['history']['valid_loss']) + 1)
        plt.plot(epochs, result['history']['valid_loss'], 
                alpha=0.6, label=result['config']['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation BPC')
    plt.title('Validation Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Best validation vs test
    plt.subplot(1, 3, 3)
    names = [r['config']['name'] for r in results]
    valid_bpc = [r['history']['best_valid_loss'] for r in results]
    test_bpc = [r['test_loss'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, valid_bpc, width, label='Valid BPC', alpha=0.8)
    plt.bar(x + width/2, test_bpc, width, label='Test BPC', alpha=0.8)
    plt.xlabel('Trial')
    plt.ylabel('BPC')
    plt.title('Best Validation vs Test BPC')
    plt.xticks(x, names, rotation=45, ha='right', fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_file}")
    plt.close()

def plot_hyperparameter_effects(results: List[Dict[str, Any]], output_dir: str):
    """Plot effect of different hyperparameters"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract hyperparameters and performance
    configs = [r['config'] for r in results]
    valid_bpc = [r['history']['best_valid_loss'] for r in results]
    
    # Learning rate effect
    ax = axes[0, 0]
    lr_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        lr = config['learning_rate']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(bpc)
    
    lrs = sorted(lr_groups.keys())
    avg_bpc = [np.mean(lr_groups[lr]) for lr in lrs]
    ax.plot(lrs, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Learning Rate Effect')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Hidden size effect
    ax = axes[0, 1]
    hs_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        hs = config['hidden_size']
        if hs not in hs_groups:
            hs_groups[hs] = []
        hs_groups[hs].append(bpc)
    
    hidden_sizes = sorted(hs_groups.keys())
    avg_bpc = [np.mean(hs_groups[hs]) for hs in hidden_sizes]
    ax.plot(hidden_sizes, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Hidden Size')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Hidden Size Effect')
    ax.grid(True, alpha=0.3)
    
    # Embedding dimension effect
    ax = axes[0, 2]
    emb_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        emb = config['embedding_dim']
        if emb not in emb_groups:
            emb_groups[emb] = []
        emb_groups[emb].append(bpc)
    
    emb_dims = sorted(emb_groups.keys())
    avg_bpc = [np.mean(emb_groups[emb]) for emb in emb_dims]
    ax.plot(emb_dims, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Embedding Dimension Effect')
    ax.grid(True, alpha=0.3)
    
    # Batch size effect
    ax = axes[1, 0]
    bs_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        bs = config['batch_size']
        if bs not in bs_groups:
            bs_groups[bs] = []
        bs_groups[bs].append(bpc)
    
    batch_sizes = sorted(bs_groups.keys())
    avg_bpc = [np.mean(bs_groups[bs]) for bs in batch_sizes]
    ax.plot(batch_sizes, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Batch Size Effect')
    ax.grid(True, alpha=0.3)
    
    # Sequence length effect
    ax = axes[1, 1]
    sl_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        sl = config['seq_len']
        if sl not in sl_groups:
            sl_groups[sl] = []
        sl_groups[sl].append(bpc)
    
    seq_lens = sorted(sl_groups.keys())
    avg_bpc = [np.mean(sl_groups[sl]) for sl in seq_lens]
    ax.plot(seq_lens, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Sequence Length Effect')
    ax.grid(True, alpha=0.3)
    
    # Number of layers effect
    ax = axes[1, 2]
    nl_groups = {}
    for config, bpc in zip(configs, valid_bpc):
        nl = config['num_layers']
        if nl not in nl_groups:
            nl_groups[nl] = []
        nl_groups[nl].append(bpc)
    
    num_layers = sorted(nl_groups.keys())
    avg_bpc = [np.mean(nl_groups[nl]) for nl in num_layers]
    ax.plot(num_layers, avg_bpc, 'o-', markersize=8)
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Avg Validation BPC')
    ax.set_title('Number of Layers Effect')
    ax.set_xticks(num_layers)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'hyperparameter_effects.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Hyperparameter effects saved to {output_file}")
    plt.close()

def print_detailed_analysis(results: List[Dict[str, Any]]):
    """Print detailed statistical analysis"""
    print("\n" + "="*80)
    print("DETAILED HYPERPARAMETER ANALYSIS")
    print("="*80)
    
    # Sort by validation loss
    results = sorted(results, key=lambda x: x['history']['best_valid_loss'])
    
    # Top 3 configurations
    print("\nTop 3 Configurations:")
    print("-" * 80)
    for i, result in enumerate(results[:3], 1):
        config = result['config']
        print(f"\n{i}. {config['name']}")
        print(f"   Valid BPC: {result['history']['best_valid_loss']:.4f}")
        print(f"   Test BPC: {result['test_loss']:.4f}")
        print(f"   Configuration:")
        print(f"     - Embedding: {config['embedding_dim']}, Hidden: {config['hidden_size']}")
        print(f"     - Layers: {config['num_layers']}, LR: {config['learning_rate']}")
        print(f"     - Batch: {config['batch_size']}, SeqLen: {config['seq_len']}")
        print(f"     - Activation: {config.get('activation', 'tanh')}")
    
    # Hyperparameter statistics
    print("\n" + "-"*80)
    print("Hyperparameter Value Analysis:")
    print("-"*80)
    
    configs = [r['config'] for r in results]
    valid_bpc = [r['history']['best_valid_loss'] for r in results]
    
    # Analyze each hyperparameter
    for param in ['learning_rate', 'hidden_size', 'embedding_dim', 'batch_size', 'seq_len', 'num_layers']:
        values = {}
        for config, bpc in zip(configs, valid_bpc):
            val = config[param]
            if val not in values:
                values[val] = []
            values[val].append(bpc)
        
        print(f"\n{param.replace('_', ' ').title()}:")
        for val in sorted(values.keys()):
            bpcs = values[val]
            print(f"  {val:>8}: avg={np.mean(bpcs):.4f}, "
                  f"min={np.min(bpcs):.4f}, max={np.max(bpcs):.4f}, "
                  f"n={len(bpcs)}")
    
    # Best values
    print("\n" + "-"*80)
    print("Best Single Values (across all trials):")
    print("-"*80)
    
    for param in ['learning_rate', 'hidden_size', 'embedding_dim', 'batch_size', 'seq_len', 'num_layers']:
        values = {}
        for config, bpc in zip(configs, valid_bpc):
            val = config[param]
            if val not in values:
                values[val] = []
            values[val].append(bpc)
        
        best_val = min(values.keys(), key=lambda v: np.mean(values[v]))
        print(f"  {param.replace('_', ' ').title()}: {best_val} "
              f"(avg BPC: {np.mean(values[best_val]):.4f})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print("\nExample:")
        print("  python analyze_results.py ../results/hyperparameter_tuning_20251007_120000")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found")
        sys.exit(1)
    
    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found in directory")
        sys.exit(1)
    
    print(f"Found {len(results)} trials")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(results, results_dir)
    plot_hyperparameter_effects(results, results_dir)
    
    # Print analysis
    print_detailed_analysis(results)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Plots saved to {results_dir}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
