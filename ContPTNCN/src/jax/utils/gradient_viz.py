"""
Gradient Visualization Utilities for JAX RNN Training

This module provides tools to analyze and visualize gradients during training.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import os


def compute_gradient_stats(grads: Dict) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each gradient tensor
    
    Args:
        grads: Gradient dictionary from JAX
    
    Returns:
        stats: Dictionary mapping parameter names to their statistics
    """
    stats = {}
    
    # Flatten the gradient tree and extract parameter names
    flat_grads = {}
    for key_path, grad in jax.tree_util.tree_leaves_with_path(grads):
        # Build parameter name from key path
        param_name = '/'.join(str(k.key) for k in key_path)
        flat_grads[param_name] = grad
    
    for name, grad in flat_grads.items():
        grad_array = np.array(grad)
        stats[name] = {
            'mean': float(np.mean(grad_array)),
            'std': float(np.std(grad_array)),
            'min': float(np.min(grad_array)),
            'max': float(np.max(grad_array)),
            'norm': float(np.linalg.norm(grad_array)),
            'shape': grad_array.shape
        }
    
    return stats


def plot_gradient_norms(grad_norms_history: Dict[str, List[float]], 
                       step_numbers: List[int],
                       save_path: str = None):
    """Plot gradient norms over training steps
    
    Args:
        grad_norms_history: Dictionary mapping parameter names to list of norms
        step_numbers: List of step numbers corresponding to the norms
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(14, 7))
    
    for param_name, norms in grad_norms_history.items():
        plt.plot(step_numbers, norms, label=param_name, marker='o', 
                markersize=3, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient Norm (L2)', fontsize=12)
    plt.title('Gradient Norms Over Training', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient norms plot to {save_path}")
    
    plt.close()


def plot_gradient_distributions(grads: Dict, step: int, save_path: str = None):
    """Plot histogram distributions of gradients for each parameter
    
    Args:
        grads: Gradient dictionary from a single step
        step: Training step number
        save_path: Path to save the plot (optional)
    """
    # Flatten gradient tree
    flat_grads = {}
    for key_path, grad in jax.tree_util.tree_leaves_with_path(grads):
        param_name = '/'.join(str(k.key) for k in key_path)
        flat_grads[param_name] = np.array(grad).flatten()
    
    # Create subplots
    n_params = len(flat_grads)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, (name, grad_vals) in enumerate(flat_grads.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Plot histogram
        ax.hist(grad_vals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(f'{name}\nMean: {np.mean(grad_vals):.2e}, Std: {np.std(grad_vals):.2e}',
                    fontsize=10)
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Gradient Distributions at Step {step}', fontsize=16, y=1.00, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient distributions to {save_path}")
    
    plt.close()


def plot_gradient_flow(grad_stats_history: List[Dict[str, Dict]], 
                       step_numbers: List[int],
                       save_path: str = None):
    """Plot gradient flow through layers over time
    
    Args:
        grad_stats_history: List of gradient statistics from multiple steps
        step_numbers: List of step numbers
        save_path: Path to save the plot (optional)
    """
    if not grad_stats_history:
        print("No gradient statistics to plot")
        return
    
    # Extract layer-wise gradient norms
    layer_norms = {}
    
    for step_stats in grad_stats_history:
        for param_name, stats in step_stats.items():
            if param_name not in layer_norms:
                layer_norms[param_name] = []
            layer_norms[param_name].append(stats['norm'])
    
    # Create plot
    plt.figure(figsize=(14, 7))
    
    # Plot each layer
    for param_name, norms in layer_norms.items():
        plt.plot(step_numbers, norms, label=param_name, marker='o', 
                markersize=3, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient Norm (L2)', fontsize=12)
    plt.title('Gradient Flow Through Network Layers', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient flow plot to {save_path}")
    
    plt.close()


def detect_gradient_issues(grad_stats: Dict[str, Dict], 
                          vanishing_threshold: float = 1e-6,
                          exploding_threshold: float = 1e3) -> Dict:
    """Detect vanishing or exploding gradients
    
    Args:
        grad_stats: Gradient statistics dictionary
        vanishing_threshold: Threshold below which gradients are considered vanishing
        exploding_threshold: Threshold above which gradients are considered exploding
    
    Returns:
        issues: Dictionary of detected gradient issues
    """
    issues = {
        'vanishing': [],
        'exploding': [],
        'normal': []
    }
    
    for param_name, stats in grad_stats.items():
        norm = stats['norm']
        
        if norm < vanishing_threshold:
            issues['vanishing'].append((param_name, norm))
        elif norm > exploding_threshold:
            issues['exploding'].append((param_name, norm))
        else:
            issues['normal'].append((param_name, norm))
    
    return issues


def print_gradient_summary(grad_stats: Dict[str, Dict], step: int):
    """Print a summary of gradient statistics
    
    Args:
        grad_stats: Gradient statistics dictionary
        step: Training step number
    """
    print(f"\n{'='*80}")
    print(f"Gradient Summary at Step {step}")
    print(f"{'='*80}")
    
    for param_name, stats in grad_stats.items():
        print(f"\n{param_name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean:  {stats['mean']:>12.6e}  |  Std:  {stats['std']:>12.6e}")
        print(f"  Min:   {stats['min']:>12.6e}  |  Max:  {stats['max']:>12.6e}")
        print(f"  Norm:  {stats['norm']:>12.6e}")
    
    # Detect issues
    issues = detect_gradient_issues(grad_stats)
    
    if issues['vanishing']:
        print(f"\n⚠️  WARNING: Vanishing gradients detected in {len(issues['vanishing'])} parameters:")
        for name, norm in issues['vanishing'][:5]:  # Show first 5
            print(f"    {name}: {norm:.2e}")
    
    if issues['exploding']:
        print(f"\n⚠️  WARNING: Exploding gradients detected in {len(issues['exploding'])} parameters:")
        for name, norm in issues['exploding'][:5]:  # Show first 5
            print(f"    {name}: {norm:.2e}")
    
    print(f"\n{'='*80}\n")


class GradientTracker:
    """Class to track gradients during training"""
    
    def __init__(self, track_interval: int = 50):
        """Initialize gradient tracker
        
        Args:
            track_interval: How often to track gradients (in steps)
        """
        self.track_interval = track_interval
        self.grad_norms_history = {}
        self.grad_stats_history = []
        self.step_numbers = []
        
    def should_track(self, step: int) -> bool:
        """Check if we should track gradients at this step"""
        return step % self.track_interval == 0
    
    def add_gradients(self, grads: Dict, step: int):
        """Add gradients for tracking
        
        Args:
            grads: Gradient dictionary
            step: Current training step
        """
        # Compute statistics
        grad_stats = compute_gradient_stats(grads)
        self.grad_stats_history.append(grad_stats)
        self.step_numbers.append(step)
        
        # Track norms for each parameter
        for param_name, stats in grad_stats.items():
            if param_name not in self.grad_norms_history:
                self.grad_norms_history[param_name] = []
            self.grad_norms_history[param_name].append(stats['norm'])
    
    def plot_all(self, output_dir: str, step: int):
        """Generate all gradient plots
        
        Args:
            output_dir: Directory to save plots
            step: Current training step
        """
        if not self.grad_stats_history:
            print("No gradient data to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot gradient norms over time
        plot_gradient_norms(
            self.grad_norms_history,
            self.step_numbers,
            save_path=os.path.join(output_dir, f'gradient_norms_step_{step}.png')
        )
        
        # Plot gradient flow
        plot_gradient_flow(
            self.grad_stats_history,
            self.step_numbers,
            save_path=os.path.join(output_dir, f'gradient_flow_step_{step}.png')
        )
        
        # Plot distributions for latest gradients
        if self.grad_stats_history:
            # Reconstruct gradient dict from latest stats
            latest_grads = {}
            for param_name in self.grad_stats_history[-1].keys():
                # Create dummy gradient for visualization (just using shape info)
                latest_grads[param_name] = np.random.randn(*self.grad_stats_history[-1][param_name]['shape']) * \
                                          self.grad_stats_history[-1][param_name]['std']
            
            plot_gradient_distributions(
                latest_grads,
                step,
                save_path=os.path.join(output_dir, f'gradient_dist_step_{step}.png')
            )
    
    def get_latest_stats(self) -> Dict:
        """Get the most recent gradient statistics"""
        if self.grad_stats_history:
            return self.grad_stats_history[-1]
        return {}
