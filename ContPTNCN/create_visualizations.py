#!/usr/bin/env python3
"""
Create comprehensive visualizations comparing Vanilla RNN vs Fast Weights RNN experiments
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_experiment_data(results_dir):
    """Load all trial data from an experiment directory"""
    results_dir = Path(results_dir)
    
    # Load summary
    with open(results_dir / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    # Load all trials
    trials = []
    for trial_file in sorted(results_dir.glob('trial_*.json')):
        with open(trial_file, 'r') as f:
            trial_data = json.load(f)
            trials.append(trial_data)
    
    return summary, trials

def create_performance_comparison(vanilla_summary, fast_summary, save_path):
    """Create side-by-side performance comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Vanilla RNN vs Fast Weights RNN: Performance Comparison\n50 Epochs, 12 Trials Each', 
                 fontsize=16, fontweight='bold')
    
    # Extract BPC values
    vanilla_bpc = [result['valid_bpc'] for result in vanilla_summary['all_results']]
    fast_bpc = [result['valid_bpc'] for result in fast_summary['all_results']]
    
    # 1. Box plot comparison
    ax1.boxplot([vanilla_bpc, fast_bpc], labels=['Vanilla RNN', 'Fast Weights RNN'])
    ax1.set_ylabel('Validation BPC')
    ax1.set_title('BPC Distribution Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add best performance annotations
    ax1.annotate(f'Best: {min(vanilla_bpc):.3f}', 
                xy=(1, min(vanilla_bpc)), xytext=(1.2, min(vanilla_bpc)),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    ax1.annotate(f'Best: {min(fast_bpc):.3f}', 
                xy=(2, min(fast_bpc)), xytext=(1.8, min(fast_bpc)),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue', fontweight='bold')
    
    # 2. Histogram comparison
    ax2.hist(vanilla_bpc, alpha=0.7, label='Vanilla RNN', bins=8, color='orange')
    ax2.hist(fast_bpc, alpha=0.7, label='Fast Weights RNN', bins=8, color='blue')
    ax2.set_xlabel('Validation BPC')
    ax2.set_ylabel('Frequency')
    ax2.set_title('BPC Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trial ranking comparison
    vanilla_ranks = list(range(1, len(vanilla_bpc) + 1))
    fast_ranks = list(range(1, len(fast_bpc) + 1))
    
    ax3.scatter(vanilla_ranks, vanilla_bpc, label='Vanilla RNN', s=100, alpha=0.7, color='orange')
    ax3.scatter(fast_ranks, fast_bpc, label='Fast Weights RNN', s=100, alpha=0.7, color='blue')
    ax3.set_xlabel('Trial Rank (by performance)')
    ax3.set_ylabel('Validation BPC')
    ax3.set_title('Trial Performance Ranking')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax4.axis('off')
    
    # Calculate statistics
    vanilla_stats = {
        'Best BPC': f"{min(vanilla_bpc):.3f}",
        'Mean BPC': f"{np.mean(vanilla_bpc):.3f}",
        'Std BPC': f"{np.std(vanilla_bpc):.3f}",
        'Median BPC': f"{np.median(vanilla_bpc):.3f}",
        '# Trials < 4.0': f"{sum(1 for x in vanilla_bpc if x < 4.0)}/12"
    }
    
    fast_stats = {
        'Best BPC': f"{min(fast_bpc):.3f}",
        'Mean BPC': f"{np.mean(fast_bpc):.3f}",
        'Std BPC': f"{np.std(fast_bpc):.3f}",
        'Median BPC': f"{np.median(fast_bpc):.3f}",
        '# Trials < 4.0': f"{sum(1 for x in fast_bpc if x < 4.0)}/12"
    }
    
    # Create table
    table_data = []
    for key in vanilla_stats:
        table_data.append([key, vanilla_stats[key], fast_stats[key]])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Metric', 'Vanilla RNN', 'Fast Weights RNN'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the best values
    for i, key in enumerate(vanilla_stats):
        if key == 'Best BPC' or key == 'Mean BPC' or key == 'Median BPC':
            if float(vanilla_stats[key].split()[0]) < float(fast_stats[key].split()[0]):
                table[(i+1, 1)].set_facecolor('#90EE90')  # Light green for better
            else:
                table[(i+1, 2)].set_facecolor('#90EE90')
    
    ax4.set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_training_curves(vanilla_trials, fast_trials, save_path):
    """Create training curves comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves: Vanilla RNN vs Fast Weights RNN', fontsize=16, fontweight='bold')
    
    # Find best trials
    vanilla_best = min(vanilla_trials, key=lambda x: x['history']['best_valid_loss'])
    fast_best = min(fast_trials, key=lambda x: x['history']['best_valid_loss'])
    
    epochs = range(1, len(vanilla_best['history']['train_loss']) + 1)
    
    # 1. Best trial training loss
    ax1.plot(epochs, vanilla_best['history']['train_loss'], 
             label=f"Vanilla RNN (Trial {vanilla_best['trial_id']})", linewidth=2, color='orange')
    ax1.plot(epochs, fast_best['history']['train_loss'], 
             label=f"Fast Weights RNN (Trial {fast_best['trial_id']})", linewidth=2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (BPC)')
    ax1.set_title('Best Trial Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Best trial validation loss
    ax2.plot(epochs, vanilla_best['history']['valid_loss'], 
             label=f"Vanilla RNN (Trial {vanilla_best['trial_id']})", linewidth=2, color='orange')
    ax2.plot(epochs, fast_best['history']['valid_loss'], 
             label=f"Fast Weights RNN (Trial {fast_best['trial_id']})", linewidth=2, color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (BPC)')
    ax2.set_title('Best Trial Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. All trials - vanilla RNN
    for trial in vanilla_trials:
        ax3.plot(epochs, trial['history']['valid_loss'], alpha=0.3, color='orange')
    # Highlight best
    ax3.plot(epochs, vanilla_best['history']['valid_loss'], 
             color='red', linewidth=3, label=f'Best (Trial {vanilla_best["trial_id"]})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Loss (BPC)')
    ax3.set_title('Vanilla RNN - All Trials')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. All trials - fast weights RNN
    for trial in fast_trials:
        ax4.plot(epochs, trial['history']['valid_loss'], alpha=0.3, color='blue')
    # Highlight best
    ax4.plot(epochs, fast_best['history']['valid_loss'], 
             color='red', linewidth=3, label=f'Best (Trial {fast_best["trial_id"]})')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss (BPC)')
    ax4.set_title('Fast Weights RNN - All Trials')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_hyperparameter_analysis(vanilla_trials, fast_trials, save_path):
    """Create hyperparameter analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
    
    # Extract hyperparameters and performance for vanilla RNN
    vanilla_data = []
    for trial in vanilla_trials:
        config = trial['config']
        vanilla_data.append({
            'hidden_size': config['hidden_size'],
            'embedding_dim': config['embedding_dim'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'seq_len': config['seq_len'],
            'num_layers': config['num_layers'],
            'valid_bpc': trial['history']['best_valid_loss']
        })
    
    # Extract hyperparameters and performance for fast weights RNN
    fast_data = []
    for trial in fast_trials:
        config = trial['config']
        fast_data.append({
            'hidden_size': config['hidden_size'],
            'embedding_dim': config['embedding_dim'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'seq_len': config['seq_len'],
            'num_layers': config['num_layers'],
            'S': config['S'],
            'valid_bpc': trial['history']['best_valid_loss']
        })
    
    # 1. Hidden size vs performance
    for model_type, data, color in [('Vanilla RNN', vanilla_data, 'orange'), ('Fast Weights', fast_data, 'blue')]:
        # Group by hidden size
        grouped = defaultdict(list)
        for item in data:
            grouped[item['hidden_size']].append(item['valid_bpc'])
        
        hidden_sizes = sorted(grouped.keys())
        means = [np.mean(grouped[hs]) for hs in hidden_sizes]
        stds = [np.std(grouped[hs]) for hs in hidden_sizes]
        
        ax1.errorbar(hidden_sizes, means, yerr=stds, 
                    label=model_type, marker='o', capsize=5, color=color)
    ax1.set_xlabel('Hidden Size')
    ax1.set_ylabel('Validation BPC')
    ax1.set_title('Hidden Size vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning rate vs performance
    for model_type, data, color in [('Vanilla RNN', vanilla_data, 'orange'), ('Fast Weights', fast_data, 'blue')]:
        # Group by learning rate
        grouped = defaultdict(list)
        for item in data:
            grouped[item['learning_rate']].append(item['valid_bpc'])
        
        learning_rates = sorted(grouped.keys())
        means = [np.mean(grouped[lr]) for lr in learning_rates]
        stds = [np.std(grouped[lr]) for lr in learning_rates]
        
        ax2.errorbar(learning_rates, means, yerr=stds, 
                    label=model_type, marker='o', capsize=5, color=color)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Validation BPC')
    ax2.set_title('Learning Rate vs Performance')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Embedding dimension vs performance
    for model_type, data, color in [('Vanilla RNN', vanilla_data, 'orange'), ('Fast Weights', fast_data, 'blue')]:
        # Group by embedding dimension
        grouped = defaultdict(list)
        for item in data:
            grouped[item['embedding_dim']].append(item['valid_bpc'])
        
        embedding_dims = sorted(grouped.keys())
        means = [np.mean(grouped[ed]) for ed in embedding_dims]
        stds = [np.std(grouped[ed]) for ed in embedding_dims]
        
        ax3.errorbar(embedding_dims, means, yerr=stds, 
                    label=model_type, marker='o', capsize=5, color=color)
    ax3.set_xlabel('Embedding Dimension')
    ax3.set_ylabel('Validation BPC')
    ax3.set_title('Embedding Dimension vs Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Fast weights S parameter analysis (only for fast weights)
    grouped_s = defaultdict(list)
    for item in fast_data:
        grouped_s[item['S']].append(item['valid_bpc'])
    
    s_values = sorted(grouped_s.keys())
    s_means = [np.mean(grouped_s[s]) for s in s_values]
    s_stds = [np.std(grouped_s[s]) for s in s_values]
    s_counts = [len(grouped_s[s]) for s in s_values]
    
    bars = ax4.bar(s_values, s_means, yerr=s_stds, 
                   capsize=5, alpha=0.7, color='blue')
    ax4.set_xlabel('S (Fast Weights Inner Steps)')
    ax4.set_ylabel('Validation BPC')
    ax4.set_title('Fast Weights S Parameter vs Performance')
    ax4.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, s_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_convergence_analysis(vanilla_trials, fast_trials, save_path):
    """Analyze convergence patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Best epoch distribution
    vanilla_best_epochs = [trial['history']['best_epoch'] for trial in vanilla_trials]
    fast_best_epochs = [trial['history']['best_epoch'] for trial in fast_trials]
    
    ax1.hist(vanilla_best_epochs, alpha=0.7, label='Vanilla RNN', bins=10, color='orange')
    ax1.hist(fast_best_epochs, alpha=0.7, label='Fast Weights RNN', bins=10, color='blue')
    ax1.set_xlabel('Best Epoch')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Best Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training time comparison
    vanilla_times = [trial['total_time']/60 for trial in vanilla_trials]  # Convert to minutes
    fast_times = [trial['total_time']/60 for trial in fast_trials]
    
    ax2.boxplot([vanilla_times, fast_times], labels=['Vanilla RNN', 'Fast Weights RNN'])
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('Training Time Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate analysis
    vanilla_final_loss = []
    vanilla_lr = []
    for trial in vanilla_trials:
        vanilla_final_loss.append(trial['history']['valid_loss'][-1])
        vanilla_lr.append(trial['config']['learning_rate'])
    
    fast_final_loss = []
    fast_lr = []
    for trial in fast_trials:
        fast_final_loss.append(trial['history']['valid_loss'][-1])
        fast_lr.append(trial['config']['learning_rate'])
    
    ax3.scatter(vanilla_lr, vanilla_final_loss, label='Vanilla RNN', s=100, alpha=0.7, color='orange')
    ax3.scatter(fast_lr, fast_final_loss, label='Fast Weights RNN', s=100, alpha=0.7, color='blue')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Final Validation BPC')
    ax3.set_title('Learning Rate vs Final Performance')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement over training
    vanilla_improvements = []
    fast_improvements = []
    
    for trial in vanilla_trials:
        initial = trial['history']['valid_loss'][0]
        final = trial['history']['best_valid_loss']
        improvement = (initial - final) / initial * 100
        vanilla_improvements.append(improvement)
    
    for trial in fast_trials:
        initial = trial['history']['valid_loss'][0]
        final = trial['history']['best_valid_loss']
        improvement = (initial - final) / initial * 100
        fast_improvements.append(improvement)
    
    ax4.boxplot([vanilla_improvements, fast_improvements], 
               labels=['Vanilla RNN', 'Fast Weights RNN'])
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Relative Improvement from Initial to Best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all visualizations"""
    # Paths
    vanilla_dir = "results/vanilla_rnn_50epochs_20251008_141726"
    fast_dir = "results/fast_weights_50epochs_20251008_141726"
    
    print("Loading experiment data...")
    vanilla_summary, vanilla_trials = load_experiment_data(vanilla_dir)
    fast_summary, fast_trials = load_experiment_data(fast_dir)
    
    print("Creating visualizations...")
    
    # Create all visualizations
    print("1/4: Performance comparison...")
    create_performance_comparison(vanilla_summary, fast_summary, 
                                'results/performance_comparison.png')
    
    print("2/4: Training curves...")
    create_training_curves(vanilla_trials, fast_trials, 
                          'results/training_curves_comparison.png')
    
    print("3/4: Hyperparameter analysis...")
    create_hyperparameter_analysis(vanilla_trials, fast_trials, 
                                  'results/hyperparameter_analysis.png')
    
    print("4/4: Convergence analysis...")
    create_convergence_analysis(vanilla_trials, fast_trials, 
                               'results/convergence_analysis.png')
    
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Generated 4 comprehensive visualization files:")
    print(f"  1. results/performance_comparison.png")
    print(f"  2. results/training_curves_comparison.png") 
    print(f"  3. results/hyperparameter_analysis.png")
    print(f"  4. results/convergence_analysis.png")
    print("="*60)
    
    # Print key findings
    vanilla_best = min(vanilla_summary['all_results'], key=lambda x: x['valid_bpc'])
    fast_best = min(fast_summary['all_results'], key=lambda x: x['valid_bpc'])
    
    print(f"\nKEY FINDINGS:")
    print(f"  Best Vanilla RNN BPC: {vanilla_best['valid_bpc']:.3f}")
    print(f"  Best Fast Weights BPC: {fast_best['valid_bpc']:.3f}")
    print(f"  Improvement: {((fast_best['valid_bpc'] - vanilla_best['valid_bpc'])/fast_best['valid_bpc']*100):.1f}% better with Vanilla RNN")
    
    avg_vanilla_time = np.mean([trial['total_time']/60 for trial in vanilla_trials])
    avg_fast_time = np.mean([trial['total_time']/60 for trial in fast_trials])
    print(f"  Avg training time - Vanilla: {avg_vanilla_time:.1f} min")
    print(f"  Avg training time - Fast Weights: {avg_fast_time:.1f} min")
    print(f"  Speed difference: {((avg_fast_time - avg_vanilla_time)/avg_vanilla_time*100):.1f}% slower with Fast Weights")

if __name__ == "__main__":
    main()