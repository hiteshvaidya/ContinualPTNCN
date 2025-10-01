#!/usr/bin/env python3
"""
Improved RNN implementation optimized for copy tasks using JAX
Features better initialization, gradient flow, and copy-specific architecture
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Dict, Any, Optional


def create_orthogonal_matrix(key: jnp.ndarray, shape: Tuple[int, int], gain: float = 1.0) -> jnp.ndarray:
    """Create an orthogonal matrix with proper scaling for RNN initialization"""
    if len(shape) != 2:
        raise ValueError("Orthogonal initialization only works for 2D matrices")
    
    rows, cols = shape
    # Generate random matrix
    random_matrix = random.normal(key, (rows, cols))
    
    # Perform QR decomposition
    q, r = jnp.linalg.qr(random_matrix)
    
    # Ensure proper scaling with gain
    d = jnp.diag(r)
    q = q * jnp.sign(d) * gain
    
    return q[:rows, :cols]


def glorot_uniform(key: jnp.ndarray, shape: Tuple[int, ...], gain: float = 1.0) -> jnp.ndarray:
    """Glorot uniform initialization for better gradient flow"""
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    limit = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, shape, minval=-limit, maxval=limit)


class CopyTaskLSTMCell:
    """LSTM Cell optimized for copy tasks with improved initialization"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def init_params(self, key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Initialize LSTM parameters with copy-task optimized initialization"""
        keys = random.split(key, 4)
        
        # Input-to-hidden weights with Xavier/Glorot initialization
        W_ih = glorot_uniform(keys[0], (4 * self.hidden_size, self.input_size), gain=1.0)
        
        # Hidden-to-hidden weights with orthogonal initialization for better gradient flow
        W_hh = create_orthogonal_matrix(keys[1], (4 * self.hidden_size, self.hidden_size), gain=1.0)
        
        # Input biases - critical for copy tasks
        b_ih = jnp.zeros(4 * self.hidden_size)
        # Forget gate bias = 3.0 for very strong memory retention (critical for copying)
        b_ih = b_ih.at[self.hidden_size:2*self.hidden_size].set(3.0)
        # Input gate bias = -1.0 to be more selective about what to remember
        b_ih = b_ih.at[0:self.hidden_size].set(-1.0)
        
        # Hidden biases
        b_hh = jnp.zeros(4 * self.hidden_size)
        
        return {
            'W_ih': W_ih,
            'W_hh': W_hh, 
            'b_ih': b_ih,
            'b_hh': b_hh
        }
    
    def forward(self, params: Dict[str, jnp.ndarray], h: jnp.ndarray, c: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through LSTM cell with numerical stability"""
        # Linear transformations - handle batched inputs
        gi = jnp.dot(x, params['W_ih'].T) + params['b_ih']  # (batch_size, 4*hidden_size)
        gh = jnp.dot(h, params['W_hh'].T) + params['b_hh']  # (batch_size, 4*hidden_size)
        
        # Split into gates: input, forget, cell, output (along the feature dimension)
        i_i, i_f, i_c, i_o = jnp.split(gi, 4, axis=-1)
        h_i, h_f, h_c, h_o = jnp.split(gh, 4, axis=-1)

        # LSTM gates with numerical stability
        input_gate = jax.nn.sigmoid(i_i + h_i)
        forget_gate = jax.nn.sigmoid(i_f + h_f)
        cell_gate = jnp.tanh(i_c + h_c)  
        output_gate = jax.nn.sigmoid(i_o + h_o)

        # Update cell state with clipping to prevent instability
        c_new = forget_gate * c + input_gate * cell_gate
        c_new = jnp.clip(c_new, -5.0, 5.0)  # Prevent exploding cell states
        
        # Update hidden state
        h_new = output_gate * jnp.tanh(c_new)

        return h_new, c_new

    def init_hidden_states(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize hidden and cell states"""
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        return h, c


class CopyTaskRNN:
    """RNN model specifically designed for copy tasks"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 output_size: int, num_layers: int = 1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Create LSTM cells
        self.cells = []
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else hidden_size
            self.cells.append(CopyTaskLSTMCell(input_size, hidden_size))
    
    def init_embedding(self, key: jnp.ndarray) -> jnp.ndarray:
        """Initialize embedding matrix"""
        return glorot_uniform(key, (self.vocab_size, self.embedding_dim), gain=1.0)
    
    def init_output_layer(self, key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Initialize output layer parameters"""
        keys = random.split(key, 2)
        W_out = glorot_uniform(keys[0], (self.output_size, self.hidden_size), gain=1.0)
        b_out = jnp.zeros(self.output_size)
        return {'W_out': W_out, 'b_out': b_out}
    
    def init_params(self, key: jnp.ndarray) -> Dict[str, Any]:
        """Initialize all model parameters"""
        keys = random.split(key, 2 + self.num_layers)
        
        params = {}
        
        # Embedding parameters
        params['embedding'] = self.init_embedding(keys[0])
        
        # LSTM layer parameters
        for i in range(self.num_layers):
            params[f'lstm_{i}'] = self.cells[i].init_params(keys[1 + i])
        
        # Output layer parameters
        params['output'] = self.init_output_layer(keys[-1])
        
        return params
    
    def embed_tokens(self, params: Dict[str, Any], token_ids: jnp.ndarray) -> jnp.ndarray:
        """Convert token IDs to embeddings"""
        return params['embedding'][token_ids]
    
    def forward_step(self, params: Dict[str, Any], hidden_states: list, cell_states: list, 
                    x: jnp.ndarray) -> Tuple[list, list, jnp.ndarray]:
        """Single forward step through all LSTM layers"""
        new_hidden_states = []
        new_cell_states = []
        
        layer_input = x
        
        for i in range(self.num_layers):
            h_new, c_new = self.cells[i].forward(
                params[f'lstm_{i}'], 
                hidden_states[i], 
                cell_states[i], 
                layer_input
            )
            new_hidden_states.append(h_new)
            new_cell_states.append(c_new)
            layer_input = h_new
        
        # Output projection - handle batched inputs
        logits = jnp.dot(layer_input, params['output']['W_out'].T) + params['output']['b_out']
        
        return new_hidden_states, new_cell_states, logits
    
    def forward_sequence_copy_task(self, params: Dict[str, Any], x_batch: jnp.ndarray) -> Tuple[jnp.ndarray, list]:
        """Forward pass for copy task sequences - x_batch contains [input_seq, padding_seq]"""
        batch_size, total_len = x_batch.shape
        seq_len = total_len // 2  # Half is input, half is padding
        
        # Initialize hidden states
        hidden_states = []
        cell_states = []
        for i in range(self.num_layers):
            h, c = self.cells[i].init_hidden_states(batch_size)
            hidden_states.append(h)
            cell_states.append(c)
        
        # Collect all logits for the entire sequence
        all_logits = []
        
        # Process the entire input sequence (both input and padding parts)
        for t in range(total_len):
            # Get embeddings for current time step
            x_t = self.embed_tokens(params, x_batch[:, t])  # (batch_size, embedding_dim)
            
            # Forward through LSTM layers
            hidden_states, cell_states, logits = self.forward_step(
                params, hidden_states, cell_states, x_t
            )
            
            all_logits.append(logits)
        
        # Stack logits: (batch_size, total_len, vocab_size)
        all_logits = jnp.stack(all_logits, axis=1)
        
        return all_logits, hidden_states


def create_copy_task_train_step(model: CopyTaskRNN, seq_len: int, padding_token: int):
    """Create training step function for copy task"""
    
    def copy_task_loss(logits: jnp.ndarray, target_indices: jnp.ndarray) -> jnp.ndarray:
        """Copy task loss - only compute loss on the copy region (second half)"""
        batch_size, total_len, vocab_size = logits.shape
        
        # Only compute loss on the second half (copy region)
        copy_region_logits = logits[:, seq_len:, :]  # (batch_size, seq_len, vocab_size)
        copy_region_targets = target_indices[:, seq_len:]  # (batch_size, seq_len)
        
        # Create mask to ignore padding tokens
        mask = (copy_region_targets != padding_token).astype(jnp.float32)
        
        # Reshape for computation
        logits_flat = copy_region_logits.reshape(-1, vocab_size)
        targets_flat = copy_region_targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # Compute cross-entropy loss
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
        
        # Apply mask and compute average loss
        masked_log_probs = target_log_probs * mask_flat
        valid_count = jnp.sum(mask_flat)
        
        loss = jnp.where(
            valid_count > 0,
            -jnp.sum(masked_log_probs) / valid_count,
            jnp.array(0.0)
        )
        
        return loss
    
    @jax.jit
    def train_step(params: Dict[str, Any], x_batch: jnp.ndarray, y_batch: jnp.ndarray, 
                  learning_rate: float = 0.001) -> Tuple[Dict[str, Any], jnp.ndarray]:
        """Single training step"""
        
        def loss_fn(params):
            logits, _ = model.forward_sequence_copy_task(params, x_batch)
            return copy_task_loss(logits, y_batch)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Gradient clipping for stability
        grads = jax.tree.map(lambda g: jnp.clip(g, -0.5, 0.5), grads)
        
        # Update parameters
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
        
        return params, loss
    
    return train_step


def create_rnn_model(vocab_size: int, embedding_dim: int, hidden_size: int, 
                    output_size: int, num_layers: int = 1, cell_type: str = 'lstm') -> CopyTaskRNN:
    """Create RNN model - simplified to only support copy task optimized LSTM"""
    if cell_type != 'lstm':
        print(f"Warning: Only LSTM supported for copy tasks, using LSTM instead of {cell_type}")
    
    return CopyTaskRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers
    )


# For backwards compatibility
RNN = CopyTaskRNN
