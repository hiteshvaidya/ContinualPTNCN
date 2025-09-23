import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from models.rnn import create_rnn_model, RNN
from utils.ptb_data_loader import PTBDataLoader
import time
import math

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for language modeling"""
    # logits: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len) - integer indices
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size * seq_len,)
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    
    # Select the log probabilities of the target tokens
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    # Return negative mean log likelihood
    return -jnp.mean(target_log_probs)

def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity"""
    return math.exp(loss)

def create_train_step_ptb(model: RNN):
    """Create a JIT-compiled training step for PTB language modeling"""
    
    @jax.jit
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, learning_rate: float = 0.001):
        """Single training step with gradient descent"""

        def loss_fn(params):
            # Forward pass through the model
            logits, _ = model.forward_sequence(params, x_batch)
            return cross_entropy_loss(logits, y_batch)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters with gradient clipping
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

        return params, loss
    
    return train_step

def evaluate_model(model: RNN, params: dict, data_loader: PTBDataLoader, dataset: str = 'valid'):
    """Evaluate model on validation or test set"""
    if dataset == 'valid':
        batches = data_loader.get_valid_batches()
    else:
        batches = data_loader.get_test_batches()
    
    total_loss = 0.0
    num_batches = 0
    
    for x_batch, y_batch in batches:
        # Forward pass
        logits, _ = model.forward_sequence(params, x_batch)
        loss = cross_entropy_loss(logits, y_batch)
        
        total_loss += loss
        num_batches += 1
        
        if num_batches >= 50:  # Limit evaluation for speed
            break
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        return float(avg_loss), perplexity(float(avg_loss))
    else:
        return float('inf'), float('inf')

def generate_text(model: RNN, params: dict, data_loader: PTBDataLoader, 
                 seed_text: str = "The", length: int = 100, temperature: float = 1.0):
    """Generate text using the trained model"""
    # Encode seed text
    seed_indices = [data_loader.char_to_idx.get(c, 0) for c in seed_text]
    
    generated_indices = seed_indices.copy()
    
    # Initialize hidden states
    states = model.init_hidden_states(1)
    
    # Process seed text first (as character indices, not one-hot)
    if len(seed_indices) > 1:
        seed_sequence = jnp.array(seed_indices[:-1]).reshape(1, -1)  # Shape: (1, seq_len)
        _, states = model.forward_sequence(params, seed_sequence, states)
    
    # Start generation from the last character of seed
    current_char_idx = seed_indices[-1]
    
    # Generate character by character
    for _ in range(length):
        # Prepare current input as character index
        current_input = jnp.array([[current_char_idx]])  # Shape: (1, 1)
        
        # Forward pass for single step
        logits, states = model.forward_sequence(params, current_input, states)
        
        # Get logits and apply temperature
        logits = logits[0, 0, :] / temperature  # (vocab_size,)
        
        # Sample from the distribution (using random sampling for better diversity)
        key = random.PRNGKey(np.random.randint(0, 10000))
        next_char_idx = int(random.categorical(key, logits, shape=(1,))[0])
        
        # Ensure valid index
        next_char_idx = max(0, min(next_char_idx, data_loader.vocab_size - 1))
        
        generated_indices.append(next_char_idx)
        current_char_idx = next_char_idx
    
    return data_loader.decode_sequence(generated_indices)

def main():
    """Main training loop for Penn Treebank"""
    print("=" * 60)
    print("JAX RNN/LSTM Training on Penn Treebank")
    print("=" * 60)
    
    # Force CPU backend for stability
    jax.config.update('jax_platform_name', 'cpu')
    print("Using CPU backend for stability")
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    print(f"Random seed: 42")
    
    # Data loading
    data_dir = "../data/ptb_char"
    print(f"Loading data from {data_dir}")
    
    try:
        data_loader = PTBDataLoader(data_dir, batch_size=20, 
                                    seq_len=35, padding=50)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the PTB data files exist in ../data/ptb_char/")
        return
    
    # Model configuration
    vocab_size = data_loader.vocab_size
    embedding_dim = 128  # Embedding dimension for character indices
    hidden_size = 256
    num_layers = 2
    cell_type = 'lstm'  # Use LSTM for better performance on long sequences
    
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
    
    # Initialize parameters
    params = model.init_params(key)
    print(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params))} parameters")
    
    # Create training function
    train_step = create_train_step_ptb(model)
    
    # Training configuration
    num_epochs = 10
    learning_rate = 0.002
    eval_every = 500
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    
    step = 0
    best_valid_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training loop
        for x_batch, y_batch in data_loader.get_train_batches():
            print(f"X: {x_batch[0]}")
            print(f"Y: {y_batch[0]}")
            exit(0)
            # Training step
            params, loss = train_step(params, x_batch, y_batch, learning_rate)
            
            epoch_loss += loss
            num_batches += 1
            step += 1
            
            # Evaluation
            if step % eval_every == 0:
                valid_loss, valid_ppl = evaluate_model(model, params, data_loader, 'valid')
                print(f"Step {step}: Train Loss = {loss:.4f}, Valid Loss = {valid_loss:.4f}, Valid PPL = {valid_ppl:.2f}")
                
                # Save best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    print(f"New best validation loss: {valid_loss:.4f}")
            
            # Limit training batches for testing
            if num_batches >= 100:
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_epoch_loss:.4f}, PPL = {perplexity(avg_epoch_loss):.2f}, Time = {epoch_time:.2f}s")
        
        # Generate sample text
        if (epoch + 1) % 2 == 0:
            sample_text = generate_text(model, params, data_loader, "The", length=50)
            print(f"Sample text: {sample_text}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    valid_loss, valid_ppl = evaluate_model(model, params, data_loader, 'valid')
    test_loss, test_ppl = evaluate_model(model, params, data_loader, 'test')
    
    print(f"Validation: Loss = {valid_loss:.4f}, Perplexity = {valid_ppl:.2f}")
    print(f"Test: Loss = {test_loss:.4f}, Perplexity = {test_ppl:.2f}")
    
    # Generate final sample
    print("\nGenerated text samples:")
    for seed in ["The", "In", "He"]:
        sample = generate_text(model, params, data_loader, seed, length=100)
        print(f"'{seed}' -> {sample}")

if __name__ == "__main__":
    main()
