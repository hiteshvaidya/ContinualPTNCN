#!/usr/bin/env python3
"""
Hyperparameter tuning for RNN on Penn Treebank character-level modeling
Supports both vanilla RNN and Fast Weights experiments with GPU selection
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
from models.rnn import create_rnn_model, RNN
from utils.ptb_data_loader import PTBDataLoader
import time
import json
import os
import sys
import argparse
from datetime import datetime
from itertools import product
from typing import Dict, Any, List, Tuple

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, task: str = 'next_char') -> jnp.ndarray:
    """Cross-entropy loss for language modeling (in bits per character)"""
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    if task == 'next_char':
        # Convert to bits per character
        log_probs = log_probs / jnp.log(2)
    
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    return -jnp.mean(target_log_probs)

def create_train_step(model: RNN):
    """Create a training step for PTB language modeling"""
    
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                   learning_rate: float = 0.001, task: str = 'next_char',
                   fast: bool = False, S: int = 0):
        """Single training step with gradient descent"""

        def loss_fn(params):
            logits, _ = model.forward_sequence(params, x_batch, task=task, 
                                               fast=fast, S=S)
            return cross_entropy_loss(logits, y_batch, task)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Gradient clipping
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

        return params, loss
    
    return train_step

def evaluate_model(model: RNN, params: dict, data_loader: PTBDataLoader, 
                   dataset: str = 'valid', max_batches: int = 20,
                   task: str = 'next_char', fast: bool = False, S: int = 0) -> float:
    """Evaluate model on validation or test set"""
    if dataset == 'valid':
        batches = data_loader.get_valid_batches(task=task)
    else:
        batches = data_loader.get_test_batches(task=task)
    
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in batches:
        logits, _ = model.forward_sequence(params, x_batch, task=task, fast=fast, S=S)
        loss = cross_entropy_loss(logits, y_batch, task)
        total_loss += loss
        num_batches += 1
        
        if num_batches >= max_batches:
            break
    
    return float(total_loss / num_batches) if num_batches > 0 else float('inf')

def train_trial(config: Dict[str, Any], data_loader: PTBDataLoader, 
                trial_id: int, results_dir: str) -> Dict[str, Any]:
    """Train a single trial with given hyperparameters"""
    
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: {config['name']}")
    print(f"{'='*60}")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(config.get('seed', 42))
    
    # Create model
    model = create_rnn_model(
        vocab_size=data_loader.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        output_size=data_loader.vocab_size,
        num_layers=config['num_layers'],
        cell_type=config.get('cell_type', 'rnn'),
        activation=config.get('activation', 'tanh')
    )
    
    # Initialize parameters
    params = model.init_params(key)
    
    # Create training function
    train_step = create_train_step(model)
    
    # Training configuration
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    seq_len = config['seq_len']
    task = config.get('task', 'next_char')
    fast_choice = config.get('fast_choice', False)
    S = config.get('S', 0)
    
    # Recreate data loader with trial-specific batch size and seq_len
    trial_data_loader = PTBDataLoader(
        data_loader.data_dir, 
        batch_size=batch_size, 
        seq_len=seq_len
    )
    
    # Training history
    history = {
        'train_loss': [],
        'valid_loss': [],
        'epoch_times': [],
        'best_valid_loss': float('inf'),
        'best_epoch': 0
    }
    
    best_params = None
    start_time = time.time()
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Fast Weights: {fast_choice}, S: {S}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training loop
        progress_bar = tqdm(trial_data_loader.get_train_batches(task=task), 
                           desc=f"Epoch {epoch+1}/{num_epochs}",
                           leave=False)
        
        for x_batch, y_batch in progress_bar:
            params, loss = train_step(params, x_batch, y_batch, learning_rate, 
                                     task, fast_choice, S)
            
            epoch_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Limit training batches to speed up tuning
            max_train_batches = config.get('max_train_batches', 200)
            if max_train_batches is not None and num_batches >= max_train_batches:
                break
        
        # Calculate average training loss
        avg_train_loss = float(epoch_loss / num_batches) if num_batches > 0 else float('inf')
        
        # Validation
        valid_loss = evaluate_model(model, params, trial_data_loader, 'valid', 
                                   max_batches=config.get('max_valid_batches', 20),
                                   task=task, fast=fast_choice, S=S)
        
        epoch_time = time.time() - epoch_start_time
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(valid_loss)
        history['epoch_times'].append(epoch_time)
        
        # Track best model
        if valid_loss < history['best_valid_loss']:
            history['best_valid_loss'] = valid_loss
            history['best_epoch'] = epoch
            best_params = params  # Save best parameters
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train BPC = {avg_train_loss:.4f}, "
              f"Valid BPC = {valid_loss:.4f}, "
              f"Time = {epoch_time:.2f}s")
        
        # Early stopping if loss explodes
        if avg_train_loss > 10.0 or np.isnan(avg_train_loss):
            print("Training diverged, stopping early")
            break
    
    total_time = time.time() - start_time
    
    # Final test evaluation with best parameters
    test_loss = evaluate_model(model, best_params if best_params is not None else params, 
                              trial_data_loader, 'test', 
                              max_batches=config.get('max_valid_batches', 20))
    
    # Compile results
    results = {
        'trial_id': trial_id,
        'config': config,
        'history': history,
        'test_loss': test_loss,
        'total_time': total_time,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_file = os.path.join(results_dir, f'trial_{trial_id:03d}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTrial {trial_id} completed:")
    print(f"  Best Valid BPC: {history['best_valid_loss']:.4f} (epoch {history['best_epoch']+1})")
    print(f"  Test BPC: {test_loss:.4f}")
    print(f"  Total time: {total_time:.2f}s")
    
    return results

def generate_trial_configurations() -> List[Dict[str, Any]]:
    """Generate different hyperparameter configurations to try"""
    
    trials = []
    
    # Base configuration
    base_config = {
        'seed': 42,
        'num_epochs': 15,
        'max_train_batches': 200,  # Limit for faster tuning
        'max_valid_batches': 20,
        'activation': 'tanh'
    }
    
    # Trial 1: Baseline (small model)
    trials.append({
        **base_config,
        'name': 'Baseline_Small',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 2: Larger embedding
    trials.append({
        **base_config,
        'name': 'Large_Embedding',
        'embedding_dim': 128,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 3: Larger hidden size
    trials.append({
        **base_config,
        'name': 'Large_Hidden',
        'embedding_dim': 64,
        'hidden_size': 256,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 4: Two layers
    trials.append({
        **base_config,
        'name': 'Two_Layers',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 5: Higher learning rate
    trials.append({
        **base_config,
        'name': 'High_LR',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.01,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 6: Lower learning rate
    trials.append({
        **base_config,
        'name': 'Low_LR',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 7: Larger batch size
    trials.append({
        **base_config,
        'name': 'Large_Batch',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'seq_len': 20
    })
    
    # Trial 8: Longer sequences
    trials.append({
        **base_config,
        'name': 'Long_Sequences',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 35
    })
    
    # Trial 9: Best combo - larger model
    trials.append({
        **base_config,
        'name': 'Best_Combo_Large',
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'learning_rate': 0.005,
        'batch_size': 64,
        'seq_len': 35
    })
    
    # Trial 10: ReLU activation
    trials.append({
        **base_config,
        'name': 'ReLU_Activation',
        'embedding_dim': 64,
        'hidden_size': 128,
        'num_layers': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_len': 20,
        'activation': 'relu'
    })
    
    # Trial 11: Medium learning rate
    trials.append({
        **base_config,
        'name': 'Medium_LR',
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 1,
        'learning_rate': 0.003,
        'batch_size': 32,
        'seq_len': 20
    })
    
    # Trial 12: Balanced configuration
    trials.append({
        **base_config,
        'name': 'Balanced',
        'embedding_dim': 96,
        'hidden_size': 192,
        'num_layers': 1,
        'learning_rate': 0.002,
        'batch_size': 48,
        'seq_len': 25
    })
    
    return trials

def summarize_results(results_dir: str):
    """Summarize all trial results and find best configuration"""
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*80}\n")
    
    # Load all results
    all_results = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith('trial_') and filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_results.append(json.load(f))
    
    if not all_results:
        print("No results found!")
        return
    
    # Sort by best validation loss
    all_results.sort(key=lambda x: x['history']['best_valid_loss'])
    
    # Print summary table
    print(f"{'Rank':<6} {'Trial':<25} {'Valid BPC':<12} {'Test BPC':<12} {'Epochs':<8} {'Time (s)':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(all_results, 1):
        config = result['config']
        print(f"{rank:<6} {config['name']:<25} "
              f"{result['history']['best_valid_loss']:<12.4f} "
              f"{result['test_loss']:<12.4f} "
              f"{result['history']['best_epoch']+1:<8} "
              f"{result['total_time']:<10.2f}")
    
    # Print best configuration details
    best = all_results[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Trial: {best['config']['name']}")
    print(f"Validation BPC: {best['history']['best_valid_loss']:.4f}")
    print(f"Test BPC: {best['test_loss']:.4f}")
    print(f"\nHyperparameters:")
    for key, value in best['config'].items():
        if key not in ['name', 'seed', 'max_train_batches', 'max_valid_batches']:
            print(f"  {key}: {value}")
    
    # Save summary
    summary_file = os.path.join(results_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'best_config': best['config'],
            'best_valid_bpc': best['history']['best_valid_loss'],
            'best_test_bpc': best['test_loss'],
            'all_results': [
                {
                    'name': r['config']['name'],
                    'valid_bpc': r['history']['best_valid_loss'],
                    'test_bpc': r['test_loss']
                }
                for r in all_results
            ]
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_trials_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate trial configurations from config file"""
    search_space = config['search_space']
    fixed_params = config.get('fixed_params', {})
    
    # Get all parameter combinations
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    
    all_combinations = list(product(*values))
    
    # Limit to num_trials if specified
    num_trials = config.get('num_trials')
    if num_trials and num_trials < len(all_combinations):
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:num_trials]
    
    # Convert to trial configs
    trials = []
    for idx, combo in enumerate(all_combinations, 1):
        trial = {keys[i]: combo[i] for i in range(len(keys))}
        trial.update(fixed_params)
        trial['name'] = f"Trial_{idx}"
        trial['num_epochs'] = config['num_epochs']
        trial['task'] = config.get('task', 'next_char')
        trial['cell_type'] = config.get('cell_type', 'rnn')
        trial['fast_choice'] = config.get('fast_choice', False)
        trial['S'] = trial.get('S', config.get('S', 0))  # Use trial-specific S or default
        trial['max_train_batches'] = config.get('max_train_batches')
        trial['max_valid_batches'] = config.get('max_valid_batches', 20)
        trials.append(trial)
    
    return trials

def main():
    """Main hyperparameter tuning routine"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RNN Hyperparameter Tuning')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RNN HYPERPARAMETER TUNING - Penn Treebank Character-Level Modeling")
    print("=" * 80)
    
    # Setup
    devices = jax.devices()
    print(f"\nAvailable devices: {devices}")
    
    # Load configuration or use defaults
    if args.config:
        print(f"\nLoading configuration from: {args.config}")
        exp_config = load_config_file(args.config)
        
        # Set GPU if specified
        gpu_id = exp_config.get('gpu_id')
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"Using GPU: {gpu_id}")
        
        data_dir = exp_config.get('data_dir', '../data/ptb_char')
        experiment_name = exp_config.get('experiment_name', 'hyperparameter_tuning')
    else:
        print("\nNo configuration file specified, using defaults")
        exp_config = None
        data_dir = "../data/ptb_char"
        experiment_name = "hyperparameter_tuning"
        
        if jax.devices('gpu'):
            print("Using GPU backend")
        else:
            print('No GPU found, using CPU backend')
            jax.config.update('jax_platform_name', 'cpu')
    
    # Load data
    print(f"\nLoading data from {data_dir}")
    
    try:
        # Use a base data loader just to get vocab info
        base_data_loader = PTBDataLoader(data_dir, batch_size=32, seq_len=20)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"../results/{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Generate trial configurations
    if exp_config:
        trials = generate_trials_from_config(exp_config)
        print(f"\nGenerated {len(trials)} trial configurations from config file")
        print(f"Experiment: {exp_config.get('description', 'N/A')}")
    else:
        trials = generate_trial_configurations()
        print(f"\nGenerated {len(trials)} trial configurations (default)")
    
    # Run trials
    all_results = []
    for i, config in enumerate(trials, 1):
        try:
            result = train_trial(config, base_data_loader, i, results_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\nError in trial {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summarize results
    print("\n" + "=" * 80)
    summarize_results(results_dir)
    print("=" * 80)
    print("\nHyperparameter tuning completed!")

if __name__ == "__main__":
    main()
