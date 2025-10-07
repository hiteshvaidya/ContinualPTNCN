#!/usr/bin/env python3
"""
Quick single trial runner for testing individual configurations
Usage: python quick_trial.py --config configs/trial_config.json
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
import argparse

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss (bits per character)"""
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1) / jnp.log(2)
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    return -jnp.mean(target_log_probs)

def create_train_step(model: RNN):
    """Create training step"""
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                   learning_rate: float = 0.001):
        def loss_fn(params):
            logits, _ = model.forward_sequence(params, x_batch, task='next_char', fast=False)
            return cross_entropy_loss(logits, y_batch)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
        return params, loss
    
    return train_step

def evaluate_model(model: RNN, params: dict, data_loader: PTBDataLoader, 
                   dataset: str = 'valid', max_batches: int = 20) -> float:
    """Evaluate model"""
    if dataset == 'valid':
        batches = data_loader.get_valid_batches(task='next_char')
    else:
        batches = data_loader.get_test_batches(task='next_char')
    
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in batches:
        logits, _ = model.forward_sequence(params, x_batch, task='next_char', fast=False)
        loss = cross_entropy_loss(logits, y_batch)
        total_loss += loss
        num_batches += 1
        
        if num_batches >= max_batches:
            break
    
    return float(total_loss / num_batches) if num_batches > 0 else float('inf')

def run_trial(config: dict):
    """Run a single trial"""
    
    print("=" * 80)
    print("RNN Training Trial")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup
    if jax.devices('gpu'):
        print("\nUsing GPU backend")
    else:
        print('\nNo GPU, using CPU backend')
        jax.config.update('jax_platform_name', 'cpu')
    
    # Load data
    data_dir = config.get('data_dir', '../data/ptb_char')
    data_loader = PTBDataLoader(
        data_dir, 
        batch_size=config['batch_size'], 
        seq_len=config['seq_len']
    )
    
    # Create model
    key = random.PRNGKey(config.get('seed', 42))
    model = create_rnn_model(
        vocab_size=data_loader.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        output_size=data_loader.vocab_size,
        num_layers=config['num_layers'],
        cell_type='rnn',
        activation=config.get('activation', 'tanh')
    )
    
    params = model.init_params(key)
    train_step = create_train_step(model)
    
    # Training
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in tqdm(data_loader.get_train_batches(task='next_char'), 
                                     desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            params, loss = train_step(params, x_batch, y_batch, learning_rate)
            epoch_loss += loss
            num_batches += 1
            
            if num_batches >= config.get('max_train_batches', 500):
                break
        
        avg_train_loss = float(epoch_loss / num_batches)
        valid_loss = evaluate_model(model, params, data_loader, 'valid', 
                                   max_batches=config.get('max_valid_batches', 20))
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train BPC = {avg_train_loss:.4f}, "
              f"Valid BPC = {valid_loss:.4f}, "
              f"Time = {epoch_time:.2f}s")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"  ✓ New best validation BPC: {valid_loss:.4f}")
    
    # Final test
    test_loss = evaluate_model(model, params, data_loader, 'test', 
                              max_batches=config.get('max_valid_batches', 20))
    
    print(f"\nFinal Results:")
    print(f"  Best Valid BPC: {best_valid_loss:.4f}")
    print(f"  Test BPC: {test_loss:.4f}")
    
    return {
        'best_valid_bpc': best_valid_loss,
        'test_bpc': test_loss
    }

def main():
    parser = argparse.ArgumentParser(description='Run a single RNN training trial')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--activation', type=str, default='tanh')
    
    args = parser.parse_args()
    
    # Load config from file or use command line args
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'num_epochs': args.num_epochs,
            'activation': args.activation,
            'seed': 42,
            'max_train_batches': 500,
            'max_valid_batches': 20
        }
    
    run_trial(config)

if __name__ == "__main__":
    main()
