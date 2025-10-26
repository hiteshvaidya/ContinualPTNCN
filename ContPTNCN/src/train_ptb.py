import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
from models.rnn import create_rnn_model, RNN, create_copy_task_train_step, copy_task_loss
from utils.ptb_data_loader import PTBDataLoader
from utils.gradient_viz import GradientTracker, print_gradient_summary
import time
import math
import os

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, task: str = None) -> jnp.ndarray:
    """Cross-entropy loss for language modeling"""
    # logits: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len) - integer indices
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size * seq_len,)
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    if task == 'next_char':
        # Convert to bits per character (divide by log(2) to get log base 2)
        log_probs = log_probs / jnp.log(2)
    
    # Select the log probabilities of the target tokens
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    # Return negative mean log likelihood
    return -jnp.mean(target_log_probs)

def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity"""
    return math.exp(loss)

def create_train_step_ptb(model: RNN):
    """Create a JIT-compiled training step for PTB language modeling"""
    
    def clip_gradients(grads, max_norm: float = 5.0):
        """Clip gradients by global norm"""
        # compute global norm
        global_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)]))

        # compute clipping coefficient
        clip_coef = max_norm / (global_norm + 1e-6)
        clip_coef = jnp.minimum(clip_coef, 1.0)

        # Apply clipping
        clipped_grads = jax.tree.map(lambda g: g * clip_coef, grads)

        return clipped_grads, global_norm

    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                   states: jnp.ndarray = None,
                   learning_rate: float = 0.001, task: str = None, 
                   fast: bool = False, S: int = 2, return_grads: bool = False):
        """Single training step with gradient descent"""

        def loss_fn(params):
            # Forward pass through the model
            logits, _ = model.forward_sequence(params, x_batch, task,
                                               initial_states=states, fast=fast, S=S)
            loss = cross_entropy_loss(logits, y_batch, task)
            
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters with gradient clipping
        # clipped_grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, clipped_grads)

        if return_grads:
            return params, loss, grads
        return params, loss
    
    # Apply jit compilation with static_argnums after function definition
    # return jax.jit(train_step, static_argnums=(4,))
    return train_step

def evaluate_model(model: RNN, params: dict, data_loader: PTBDataLoader, dataset: str = 'valid', task: str = 'next_char'):
    """Evaluate model on validation or test set for copy task"""
    if dataset == 'valid':
        batches = data_loader.get_valid_batches(task)
    else:
        batches = data_loader.get_test_batches(task)
    
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in batches:
        # Forward pass for copy task
        logits, _ = model.forward_sequence(params, x_batch, task)
        if task == 'copy':
            loss = copy_task_loss(logits, y_batch, data_loader.seq_len, data_loader.padding)
        else:
            loss = cross_entropy_loss(logits, y_batch, task)

        total_loss += loss
        num_batches += 1
        
        if num_batches >= 10:  # Limit evaluation for speed
            break
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        return float(avg_loss), perplexity(float(avg_loss))
    else:
        return float('inf'), float('inf')

def generate_text(model: RNN, params: dict, data_loader: PTBDataLoader, 
                 seed_text: str = "The University of South Florida", length: int = 100, temperature: float = 1.0, task: str = None):
    """Generate text using the trained model"""
    # Encode seed text
    seed_text = seed_text.lower().replace(" ", "")
    input_length = len(seed_text)
    print(f"seed text: {seed_text}")
    seed_indices = [data_loader.char_to_idx.get(c, 0) for c in seed_text]

    if task == 'copy':
        seed_indices = seed_indices + [data_loader.padding] * input_length
        output_indices = [data_loader.padding] * input_length + [data_loader.char_to_idx.get(c, 0) for c in seed_text]
    elif task == 'next_char':
        output_indices = seed_indices.copy()
    else:
        raise ValueError("Task must be 'copy' or 'next_char'")

    generated_indices = seed_indices.copy()
    
    # Initialize hidden states
    states = model.init_hidden_states(1)
    
    # Process seed text to warm up the model
    if len(seed_indices) > 1:
        seed_sequence = jnp.array(seed_indices[:-1]).reshape(1, -1)  # All but last char
        _, states = model.forward_sequence(params, seed_sequence, task, states)
    
    # Start generation from the last character of seed
    current_char_idx = seed_indices[-1] if seed_indices else 0
    generated_indices = seed_indices.copy()
    
    # Generate character by character
    for i in range(length):
        # Forward pass with single character
        current_input = jnp.array([[current_char_idx]])
        logits, states = model.forward_sequence(params, current_input, task, states)
        
        # Get next character prediction
        next_char_logits = logits[0, -1, :]  # Last timestep, first batch
        
        # Apply temperature and sample
        scaled_logits = next_char_logits / temperature
        probs = jax.nn.softmax(scaled_logits)
        
        # Sample from distribution (or use argmax for deterministic)
        key = random.PRNGKey(42 + i)  # Use step-based key for reproducibility
        next_char_idx = random.categorical(key, jnp.log(probs + 1e-8))  # Add small epsilon for stability
        
        generated_indices.append(int(next_char_idx))
        current_char_idx = int(next_char_idx)
        
        # Stop if we generate end token or reach max length
        if next_char_idx == data_loader.char_to_idx.get('<eos>', -1):
            break
    
    return data_loader.decode_sequence(generated_indices)

def predict(logits: jnp.ndarray, data_loader: PTBDataLoader, temperature: float = 1.0) -> int:
    """Predict next character index from logits"""
    # Apply temperature
    scaled_logits = logits / temperature
    probs = jax.nn.softmax(scaled_logits)
    probs = jnp.argmax(probs, axis=-1)
    probs = jnp.squeeze(probs).tolist()
    return probs

def evaluate_copy_accuracy(model: RNN, params: dict, data_loader: PTBDataLoader, num_examples: int = 5, task: str = None):
    """Evaluate copy task accuracy"""
    correct_copies = 0
    total_examples = 0
    
    batches = data_loader.get_valid_batches()
    
    for x_batch, y_batch in batches:
        # Get predictions
        logits, _ = model.forward_sequence(params, x_batch, task=task)
        predictions = jnp.argmax(logits, axis=-1)  # (batch_size, 2*seq_len)
        
        batch_size = x_batch.shape[0]
        seq_len = data_loader.seq_len
        
        for i in range(min(batch_size, num_examples)):
            # Input sequence (first half)
            input_seq = x_batch[i, :seq_len]
            # Target sequence (second half)  
            target_seq = y_batch[i, seq_len:]
            # Predicted sequence (second half)
            pred_seq = predictions[i, seq_len:]
            
            # Check if prediction matches target (ignoring padding)
            mask = target_seq != data_loader.padding
            if jnp.sum(mask) > 0:  # Only count non-empty sequences
                correct = jnp.all(pred_seq[mask] == target_seq[mask])
                correct_copies += int(correct)
                total_examples += 1
                
                if total_examples <= 3:  # Show first few examples
                    print(f"Example {total_examples}:")
                    # Convert JAX arrays to numpy arrays first
                    input_masked = jnp.array(input_seq)[mask]
                    target_masked = jnp.array(target_seq)[mask] 
                    pred_masked = jnp.array(pred_seq)[mask]
                    
                    print(f"  Input:  {data_loader.decode_sequence(np.array(input_masked))}")
                    print(f"  Target: {data_loader.decode_sequence(np.array(target_masked))}")  
                    print(f"  Pred:   {data_loader.decode_sequence(np.array(pred_masked))}")
                    print(f"  Match:  {correct}")
        
        total_examples = min(total_examples, num_examples)
        if total_examples >= num_examples:
            break
    
    accuracy = correct_copies / total_examples if total_examples > 0 else 0.0
    return accuracy, correct_copies, total_examples


def main():
    """Main training loop for Penn Treebank"""
    print("=" * 60)
    print("JAX RNN/LSTM Training on Penn Treebank")
    print("=" * 60)
    
    # Check available devices and use GPU if available
    devices = jax.devices()
    print(f"Available devices: {devices}")

    if jax.devices('gpu'):
        print("Using GPU backend")
    else:
        print('No GPU found, using CPU backend')
        jax.config.update('jax_platform_name', 'cpu')
    
    # Print current backend
    print(f"JAX backend: {jax.default_backend()}")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    print(f"Random seed: 42")
    
    # Data loading
    data_dir = "../data/ptb_char"
    print(f"Loading data from {data_dir}")
    
    try:
        data_loader = PTBDataLoader(data_dir, batch_size=64, 
                                    seq_len=10)  # Let it auto-detect padding token
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the PTB data files exist in ../data/ptb_char/")
        return
    
    # Model configuration
    vocab_size = data_loader.vocab_size
    embedding_dim = 128  # Embedding dimension for character indices
    hidden_size = 256
    num_layers = 1
    cell_type = 'rnn'  # Use LSTM for better performance on long sequences
    task = 'next_char'
    fast_choice = True
    S = 3

    print(f"Creating {cell_type.upper()} model:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    
    # Create model for character indices (not one-hot)
    model = create_rnn_model(
        vocab_size=vocab_size,  # Size of character vocabulary
        embedding_dim=embedding_dim,  # Embedding dimension
        hidden_size=hidden_size,
        output_size=vocab_size,  # Predict next character
        num_layers=num_layers,
        cell_type=cell_type
    )
    
    # Initialize parameters (with fast weights if enabled)
    params = model.init_params(key, use_fast_weights=fast_choice)
    # print(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params))} parameters")
    
    # Create training function
    train_step = create_train_step_ptb(model)
    # train_step = create_copy_task_train_step(model, data_loader.seq_len, data_loader.padding)
    
    # Training configuration
    num_epochs = 25
    learning_rate = 0.01  # Higher learning rate for copy task
    eval_every = 200  # More frequent evaluation to track progress
    
    # Gradient visualization setup
    gradient_tracker = GradientTracker(track_interval=50)
    gradient_output_dir = "gradient_plots"
    os.makedirs(gradient_output_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient plots will be saved to: {gradient_output_dir}")
    
    step = 0
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
        # Training loop
        for x_batch, y_batch in tqdm(data_loader.get_train_batches(task)):
            states = model.init_hidden_states(data_loader.batch_size,
                                              fast_weights=fast_choice,
                                              )
            
            # Track gradients at regular intervals
            should_track_grads = gradient_tracker.should_track(step)
            
            # Training step (with optional gradient return)
            if should_track_grads:
                params, loss, grads = train_step(params, x_batch, y_batch, 
                                                states=states,
                                                learning_rate=learning_rate, task=task,
                                                fast=fast_choice, S=S, return_grads=True)
                # Track gradients
                gradient_tracker.add_gradients(grads, step)
                
                # Print gradient summary every 200 steps
                if step % 200 == 0:
                    print_gradient_summary(gradient_tracker.get_latest_stats(), step)
            else:
                params, loss = train_step(params, x_batch, y_batch, 
                                         states=states,
                                         learning_rate=learning_rate, task=task,
                                         fast=fast_choice, S=S, return_grads=False)
            
            epoch_loss += loss
            num_batches += 1
            step += 1
            
            # Evaluation and gradient plotting
            if step % eval_every == 0:
                valid_loss, valid_ppl = evaluate_model(model, params, data_loader, 'valid', task=task)
                print(f"Step {step}: Train BPC = {loss:.4f}, Valid BPC = {valid_loss:.4f}, Valid PPL = {valid_ppl:.2f}")
                
                # Generate gradient plots every 1000 steps
                if step % 1000 == 0 and len(gradient_tracker.step_numbers) > 5:
                    print(f"\nGenerating gradient plots at step {step}...")
                    gradient_tracker.plot_all(gradient_output_dir, step)
                    print(f"Gradient plots saved to {gradient_output_dir}/\n")
                
                # Save best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    print(f"New best validation BPC: {valid_loss:.4f}")
            
            # Limit training batches for testing
            if num_batches >= 100:
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch + 1}/{num_epochs}: Avg BPC = {avg_epoch_loss:.4f}, Time = {epoch_time:.2f}s") # PPL = {perplexity(avg_epoch_loss):.2f}
        
        # Generate sample text
        if (epoch + 1) % 2 == 0:
            sample_text = generate_text(model, params, data_loader, "The", length=50, task=task)
            print(f"Sample text: {sample_text}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    valid_loss, valid_ppl = evaluate_model(model, params, data_loader, 'valid', task)
    test_loss, test_ppl = evaluate_model(model, params, data_loader, 'test', task)
    
    print(f"Validation: Loss = {valid_loss:.4f}, Perplexity = {valid_ppl:.2f}")
    print(f"Test: Loss = {test_loss:.4f}, Perplexity = {test_ppl:.2f}")
    
    # Generate final gradient plots
    if len(gradient_tracker.step_numbers) > 0:
        print("\nGenerating final gradient analysis...")
        gradient_tracker.plot_all(gradient_output_dir, step)
        print(f"Final gradient plots saved to {gradient_output_dir}/")
        
        # Print final gradient summary
        print("\nFinal Gradient Statistics:")
        print_gradient_summary(gradient_tracker.get_latest_stats(), step)
    
    # Generate final sample
    print("\nGenerated text samples:")
    for seed in ["The", "In", "He"]:
        sample = generate_text(model, params, data_loader, seed, length=100, task=task)
        print(f"'{seed}' -> {sample}")

if __name__ == "__main__":
    main()
