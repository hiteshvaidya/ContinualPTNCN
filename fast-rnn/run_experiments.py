#!/usr/bin/env python3
"""
Experiment Runner and Logger for Fast Weights
Runs experiments for both RNN and LSTM implementations and logs results
"""

import subprocess
import json
import os
import datetime
import argparse
from pathlib import Path


class ExperimentLogger:
    """Logger for tracking experiments"""
    
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"experiments_{self.timestamp}.json"
        self.experiments = []
    
    def log_experiment(self, exp_config, result):
        """Log a single experiment"""
        experiment = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": exp_config,
            "result": result
        }
        self.experiments.append(experiment)
        self._save()
    
    def _save(self):
        """Save experiments to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump({
                "session_start": self.timestamp,
                "experiments": self.experiments
            }, f, indent=2)
        print(f"Logged to: {self.log_file}")
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for i, exp in enumerate(self.experiments, 1):
            print(f"\n--- Experiment {i} ---")
            print(f"Model: {exp['config']['model_type']} - {exp['config']['model']}")
            print(f"Dataset: {exp['config']['dataset']}")
            print(f"Config: {exp['config']}")
            
            result = exp['result']
            if result['success']:
                print(f"✓ SUCCESS")
                print(f"  Best Valid BPC: {result.get('best_valid_bpc', 'N/A')}")
                print(f"  Test BPC: {result.get('test_bpc', 'N/A')}")
                print(f"  Parameters: {result.get('num_params', 'N/A')}")
                print(f"  Checkpoint: {result.get('checkpoint', 'N/A')}")
            else:
                print(f"✗ FAILED")
                print(f"  Error: {result.get('error', 'Unknown error')}")


def run_experiment(script, args_dict, exp_name, logger):
    """Run a single experiment"""
    
    # Build command
    cmd = ["python", script]
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print("\n" + "="*80)
    print(f"Running: {exp_name}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Parse output for metrics
        output = result.stdout
        
        # Extract metrics from output
        best_valid_bpc = None
        test_bpc = None
        num_params = None
        
        for line in output.split('\n'):
            if "Best valid BPC:" in line or "Saved best model with valid BPC:" in line:
                try:
                    best_valid_bpc = float(line.split(':')[-1].strip())
                except:
                    pass
            elif "Test BPC:" in line:
                try:
                    test_bpc = float(line.split(':')[-1].strip())
                except:
                    pass
            elif "Number of parameters:" in line:
                try:
                    num_params = line.split(':')[-1].strip()
                except:
                    pass
        
        # Log result
        exp_result = {
            "success": result.returncode == 0,
            "best_valid_bpc": best_valid_bpc,
            "test_bpc": test_bpc,
            "num_params": num_params,
            "checkpoint": args_dict.get('save_path', 'N/A'),
            "stdout": output,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
        logger.log_experiment({
            "name": exp_name,
            "script": script,
            "model_type": "RNN" if "fast_weights.py" in script else "LSTM",
            **args_dict
        }, exp_result)
        
        if result.returncode == 0:
            print(f"✓ {exp_name} completed successfully")
            if best_valid_bpc:
                print(f"  Best Valid BPC: {best_valid_bpc:.4f}")
            if test_bpc:
                print(f"  Test BPC: {test_bpc:.4f}")
        else:
            print(f"✗ {exp_name} failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
        
        return exp_result
        
    except subprocess.TimeoutExpired:
        error_msg = f"Experiment timed out after 2 hours"
        print(f"✗ {error_msg}")
        
        exp_result = {
            "success": False,
            "error": error_msg
        }
        
        logger.log_experiment({
            "name": exp_name,
            "script": script,
            "model_type": "RNN" if "fast_weights.py" in script else "LSTM",
            **args_dict
        }, exp_result)
        
        return exp_result
    
    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}"
        print(f"✗ {error_msg}")
        
        exp_result = {
            "success": False,
            "error": error_msg
        }
        
        logger.log_experiment({
            "name": exp_name,
            "script": script,
            "model_type": "RNN" if "fast_weights.py" in script else "LSTM",
            **args_dict
        }, exp_result)
        
        return exp_result


def main():
    parser = argparse.ArgumentParser(description='Run and log Fast Weights experiments')
    parser.add_argument('--dataset', type=str, default='text8', choices=['text8', 'ptb'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='data/text8',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiments (fewer epochs)')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['standard_rnn', 'fast_rnn', 'lstm', 'fast_lstm', 'all'],
                       default=['all'],
                       help='Models to run')
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Initialize logger
    logger = ExperimentLogger()
    
    # Determine which models to run
    if 'all' in args.models:
        models_to_run = ['standard_rnn', 'fast_rnn', 'lstm', 'fast_lstm']
    else:
        models_to_run = args.models
    
    # Adjust epochs for quick mode
    epochs = 3 if args.quick else args.epochs
    
    # Base configuration
    base_config = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'epochs': epochs,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'seq_length': 100,
        'lr': 0.001,
        'clip_grad': 5.0,
    }
    
    print("\n" + "="*80)
    print("FAST WEIGHTS EXPERIMENT SUITE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"GPU: {args.gpu}")
    print(f"Models: {', '.join(models_to_run)}")
    print("="*80)
    
    # Run RNN experiments
    if 'standard_rnn' in models_to_run:
        print("\n### RNN EXPERIMENTS ###\n")
        
        # Standard RNN baseline
        run_experiment(
            "fast_weights.py",
            {
                **base_config,
                'model': 'standard_rnn',
                'save_path': f'checkpoints/standard_rnn_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Standard RNN - {args.dataset}",
            logger
        )
    
    if 'fast_rnn' in models_to_run:
        # Fast Weights RNN (S=1)
        run_experiment(
            "fast_weights.py",
            {
                **base_config,
                'model': 'fast_rnn',
                'S': 1,
                'lambda_decay': 0.95,
                'eta_lr': 0.5,
                'use_layer_norm': True,
                'save_path': f'checkpoints/fast_rnn_S1_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Fast Weights RNN (S=1) - {args.dataset}",
            logger
        )
        
        # Fast Weights RNN (S=2)
        run_experiment(
            "fast_weights.py",
            {
                **base_config,
                'model': 'fast_rnn',
                'S': 2,
                'lambda_decay': 0.95,
                'eta_lr': 0.5,
                'use_layer_norm': True,
                'save_path': f'checkpoints/fast_rnn_S2_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Fast Weights RNN (S=2) - {args.dataset}",
            logger
        )
    
    # Run LSTM experiments
    if 'lstm' in models_to_run:
        print("\n### LSTM EXPERIMENTS ###\n")
        
        # Standard LSTM baseline
        run_experiment(
            "fast_weights_lstm.py",
            {
                **base_config,
                'model': 'lstm',
                'num_layers': 2,
                'dropout': 0.2,
                'save_path': f'checkpoints/lstm_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Standard LSTM - {args.dataset}",
            logger
        )
    
    if 'fast_lstm' in models_to_run:
        # Fast Weights LSTM (S=1)
        run_experiment(
            "fast_weights_lstm.py",
            {
                **base_config,
                'model': 'fast_lstm',
                'num_layers': 2,
                'dropout': 0.2,
                'S': 1,
                'lambda_decay': 0.95,
                'eta_lr': 0.5,
                'use_layer_norm': True,
                'save_path': f'checkpoints/fast_lstm_S1_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Fast Weights LSTM (S=1) - {args.dataset}",
            logger
        )
        
        # Fast Weights LSTM (S=2)
        run_experiment(
            "fast_weights_lstm.py",
            {
                **base_config,
                'model': 'fast_lstm',
                'num_layers': 2,
                'dropout': 0.2,
                'S': 2,
                'lambda_decay': 0.95,
                'eta_lr': 0.3,  # Lower eta for stability with S=2
                'use_layer_norm': True,
                'save_path': f'checkpoints/fast_lstm_S2_{args.dataset}_{logger.timestamp}.pt'
            },
            f"Fast Weights LSTM (S=2) - {args.dataset}",
            logger
        )
    
    # Print summary
    logger.print_summary()
    
    print("\n" + "="*80)
    print(f"All experiments completed. Results saved to: {logger.log_file}")
    print("="*80)


if __name__ == '__main__':
    main()
