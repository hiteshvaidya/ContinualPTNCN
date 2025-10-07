import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional, Callable
import functools

class RNNCell:
    """Basic RNN Cell implementation in JAX
    
    Uses Xavier/Glorot initialization for input-to-hidden weights and
    orthogonal initialization for hidden-to-hidden weights for better
    gradient flow and training stability.
    """
    def __init__(self, hidden_size: int, input_size: int, activation: str = 'tanh'):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = self._get_activation(activation)
        self.lr_lambda = 0.9
        self.lr_eta = 0.1
        self.t = 0
        self.hidden_states = []

    def _increment_time(self):
        self.t += 1

    def _get_activation(self, activation: str) -> Callable:
        """Get activation function"""
        activations = {
            'tanh': jnp.tanh,
            'relu': jax.nn.relu,
            'sigmoid': jax.nn.sigmoid,
            'identity': lambda x: x
        }
        return activations.get(activation, jnp.tanh)
    
    def init_params(self, key: jax.random.PRNGKey) -> dict:
        """initialize RNN parameters"""
        k1, k2, k3 = random.split(key, 3)

        # Xavier/Glorot initialization for input-to-hidden weights
        w_ih_std = jnp.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # Orthogonal initialization for hidden-to-hidden weights
        W_hh_init = random.normal(k2, (self.hidden_size, self.hidden_size))
        # W_hh_orthogonal = self._orthogonal_init(W_hh_init)

        params = {
            'W_ih': random.normal(k1, (self.hidden_size, self.input_size)) * w_ih_std,
            'W_hh': W_hh_init,
            'b_h': jnp.zeros((self.hidden_size)),
            # 'hidden_states': [],
        }
        return params
    
    def _orthogonal_init(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Orthogonal initialization using QR decomposition"""
        q, r = jnp.linalg.qr(matrix)
        # Make sure the diagonal of R is positive
        d = jnp.diag(r)
        q = q * jnp.sign(d)
        return q
    
    def init_hidden(self, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Initialize hidden state"""
        return jnp.zeros((batch_size, self.hidden_size)) # seq_len

    def fast_forward(self, params: dict, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Fast forward pass for multiple time steps"""
        # h0(t+1) -> standard RNN update
        h_next = self.activation(
                jnp.dot(x, params['W_ih'].T) +
                jnp.dot(h[:,self.t,:], params['W_hh'].T)
            )
        h_s_next = h_next.copy()
        h_fast = jnp.zeros_like(h_next)
        
        # h_s(t+1) -> fast weights update
        for s in range(2):
            # A(t)h_s(t+1)
            for tau in range(1, self.t):
                temp = jnp.dot(jnp.transpose(h[:,tau,:]), 
                                h_next)
                h_fast += self.lr_lambda**(self.t - tau) * jnp.dot(
                                                                h[:,tau,:], 
                                                                temp
                                                                )
            h_fast = self.lr_eta * h_fast
            
            # h_s+1(t+1) = f([Wh(t) + Cx(t)]) + A(t)h_s(t+1))
            h_s_next = h_next + h_fast

        # h[:,self.t+1,:] = h_s_next
        h = h.at[:,self.t,:].set(h_s_next)
        return h_s_next

    def __call__(self, params: dict, x: jnp.ndarray, 
                 h: jnp.ndarray, fast=False) -> jnp.ndarray:
        """forward pass for one time step"""
        if fast:
            return self.fast_forward(params, x, h)
        else:
            h_new = self.activation(
                jnp.dot(x, params['W_ih'].T) +
                jnp.dot(h, params['W_hh'].T) +
                params['b_h']
                )
            return h_new
    
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


class LSTMCell:
    """Improved LSTM Cell with better initialization and copy task optimization"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def init_params(self, key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Initialize LSTM parameters with improved initialization"""
        keys = random.split(key, 6)
        
        # Input-to-hidden weights with Glorot initialization
        W_ih = glorot_uniform(keys[0], (4 * self.hidden_size, self.input_size), gain=1.0)
        
        # Hidden-to-hidden weights with orthogonal initialization
        W_hh = create_orthogonal_matrix(keys[1], (4 * self.hidden_size, self.hidden_size), gain=1.0)
        
        # Input biases
        b_ih = jnp.zeros(4 * self.hidden_size)
        # Set forget gate bias to 2.0 for even better gradient flow in copy tasks
        b_ih = b_ih.at[self.hidden_size:2*self.hidden_size].set(2.0)
        
        # Hidden biases (small positive bias for input gate to encourage learning)
        b_hh = jnp.zeros(4 * self.hidden_size)
        b_hh = b_hh.at[0:self.hidden_size].set(0.1)  # Small positive bias for input gate
        
        return {
            'W_ih': W_ih,
            'W_hh': W_hh, 
            'b_ih': b_ih,
            'b_hh': b_hh
        }
    
    def init_hidden(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize hidden and cell states for LSTM"""
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        return (h, c)
    
    def __call__(self, params: dict, x: jnp.ndarray, state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for LSTM cell"""
        h, c = state
        
        # Compute all gates at once
        gates_ih = jnp.dot(x, params['W_ih'].T) + params['b_ih']
        gates_hh = jnp.dot(h, params['W_hh'].T) + params['b_hh']
        gates = gates_ih + gates_hh
        
        # Split into individual gates
        i_gate = jax.nn.sigmoid(gates[:, :self.hidden_size])                    # Input gate
        f_gate = jax.nn.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Forget gate
        g_gate = jnp.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])      # New gate
        o_gate = jax.nn.sigmoid(gates[:, 3*self.hidden_size:])                  # Output gate
        
        # Update cell state
        c_new = f_gate * c + i_gate * g_gate
        
        # Update hidden state
        h_new = o_gate * jnp.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
class RNN:
    """Multi-layer RNN implementation in JAX for character-level language modeling"""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, cell_type: str = 'rnn', activation: str = 'tanh'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # Initialize cells - first layer takes embedded input, others take hidden states
        if cell_type == 'lstm':
            self.cells = [LSTMCell(embedding_dim if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        else:
            self.cells = [RNNCell(hidden_size, embedding_dim if i == 0 else hidden_size, activation) for i in range(num_layers)]

        
    def init_params(self, key: jax.random.PRNGKey) -> dict:
        """Initialize all parameters including embedding layer"""
        keys = random.split(key, self.num_layers + 3)

        params = {}

        # Embedding layer for character indices
        embedding_std = jnp.sqrt(1.0 / self.embedding_dim)
        params['embedding'] = random.normal(keys[0], (self.vocab_size, self.embedding_dim)) * embedding_std

        # Initialize cell parameters
        for i, cell in enumerate(self.cells):
            params[f'layer_{i}'] = cell.init_params(keys[i + 1])

        # Output layer for character prediction (vocab_size classes)
        w_out_std = jnp.sqrt(2.0 / (self.hidden_size + self.vocab_size))
        params['W_out'] = random.normal(keys[-1], (self.vocab_size, self.hidden_size)) * w_out_std
        params['b_out'] = jnp.zeros((self.vocab_size,))

        return params
    
    def init_hidden_states(self, batch_size: int, seq_len: int):
        """Initialize hidden states for all layers"""
        # if self.cell_type == 'lstm':
        #     return [cell.init_hidden(batch_size) for cell in self.cells]
        # else:
        return tuple(cell.init_hidden(batch_size, seq_len) for cell in self.cells)
    
    def forward_step(self, params: dict, x: jnp.ndarray, 
                     states,
                     fast: bool=False, t: int = None):
        """Forward pass for one time step"""
        current_input = x
        new_states = []

        # Process each layer sequentially
        for i in range(len(self.cells)):
            cell = self.cells[i]
            state = states[i]
            layer_params = params[f'layer_{i}']

            if self.cell_type == 'lstm':
                h_new, state_new = cell(layer_params, current_input, state)
                new_states.append(state_new)
                current_input = h_new
            else:
                h_new = cell(layer_params, current_input, state, fast)
                new_states.append(h_new)
                current_input = h_new
                cell._increment_time()  # Increment time step for fast weights

        # Output layer
        output = jnp.dot(current_input, params['W_out'].T) + params['b_out']

        return output, tuple(new_states)
    
    def forward_sequence(self, params: dict, x_seq: jnp.ndarray, 
                         task: str, initial_states=None, fast=False):
        """Forward pass for character index sequences"""
        batch_size, seq_len = x_seq.shape  # x_seq contains character indices

        # Convert indices to embeddings
        embedded_seq = params['embedding'][x_seq]  # Shape: (batch_size, seq_len, embedding_dim)

        if initial_states is None:
            init_states = self.init_hidden_states(batch_size, seq_len)
        else:
            init_states = initial_states

        outputs = []

        # Process sequence timestep by timestep
        for t in range(seq_len):
            current_input = embedded_seq[:, t, :]  # Shape: (batch_size, embedding_dim)
            output, states = self.forward_step(params, current_input, init_states, fast)
            outputs.append(output)

        
        return jnp.stack(outputs, axis=1), states  # Shape: (batch_size, seq_len, vocab_size)
    
    def forward_sequence_copy_task(self, params: dict, x_seq: jnp.ndarray, initial_states=None):
        """Forward pass specifically designed for copy task
        
        Args:
            x_seq: (batch_size, 2*seq_len) where first half is input, second half is padding
        
        Returns:
            outputs: (batch_size, 2*seq_len, vocab_size) where second half should match first half
        """
        batch_size, total_seq_len = x_seq.shape
        
        # Convert indices to embeddings
        embedded_seq = params['embedding'][x_seq]  # Shape: (batch_size, 2*seq_len, embedding_dim)

        if initial_states is None:
            states = self.init_hidden_states(batch_size)
        else:
            states = initial_states

        outputs = []

        # Process entire sequence (input + padding region)
        for t in range(total_seq_len):
            output, states = self.forward_step(params, embedded_seq[:, t, :], states)
            outputs.append(output)

        return jnp.stack(outputs, axis=1), states  # Shape: (batch_size, 2*seq_len, vocab_size)
    
# Utility functions
def create_rnn_model(vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int,
                     num_layers: int = 1, cell_type: str = 'rnn', activation: str = 'tanh'):
    """Factory function to create RNN model for character-level modeling"""
    return RNN(vocab_size, embedding_dim, hidden_size, output_size, num_layers, cell_type, activation)

# Example usage and training utilities
def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss"""
    return jnp.mean((predictions - targets) ** 2)

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for classification"""
    return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

def character_prediction_loss(logits: jnp.ndarray, target_indices: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for character prediction from indices"""
    # logits: (batch_size, seq_len, vocab_size)
    # target_indices: (batch_size, seq_len)
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = target_indices.reshape(-1)  # (batch_size * seq_len,)
    
    # Compute log probabilities using log base 2 for bits per character calculation
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    
    # Select the log probabilities of the target tokens
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    # Return negative mean log likelihood
    return -jnp.mean(target_log_probs)

def copy_task_loss(logits: jnp.ndarray, target_indices: jnp.ndarray, seq_len: int, padding_token: int) -> jnp.ndarray:
    """Cross-entropy loss specifically for copy task
    
    Args:
        logits: (batch_size, 2*seq_len, vocab_size) - model predictions
        target_indices: (batch_size, 2*seq_len) - target sequence ([pad, pad, ..., tokens])
        seq_len: length of original sequence to copy
        padding_token: token used for padding
    """
    batch_size, total_len, vocab_size = logits.shape
    
    # Only compute loss on the second half (copy region)
    copy_region_logits = logits[:, seq_len:, :]  # (batch_size, seq_len, vocab_size)
    copy_region_targets = target_indices[:, seq_len:]  # (batch_size, seq_len)
    
    # Create mask to ignore padding tokens in loss computation
    mask = (copy_region_targets != padding_token).astype(jnp.float32)
    
    # Reshape for computation
    logits_flat = copy_region_logits.reshape(-1, vocab_size)
    targets_flat = copy_region_targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    
    # Select log probabilities of target tokens
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    # Apply mask and compute masked mean
    masked_log_probs = target_log_probs * mask_flat
    
    # Compute valid count and use conditional logic compatible with JAX
    valid_count = jnp.sum(mask_flat)
    
    # Use jnp.where instead of if statement for JAX compatibility
    base_loss = jnp.where(
        valid_count > 0,
        -jnp.sum(masked_log_probs) / valid_count,
        jnp.array(10.0)  # High penalty when no valid tokens
    )
    
    return base_loss

def create_train_step(model: RNN, task: str = 'next_char'):
    """Create a JIT-compiled training step for character prediction"""
    
    @jax.jit
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, learning_rate: float = 0.001):
        """Single training step with gradient descent for character prediction"""

        def loss_fn(params):
            logits, _ = model.forward_sequence(params, x_batch, task=task)
            return character_prediction_loss(logits, y_batch)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Gradient clipping for stability
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)

        # Update parameters
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

        return params, loss
    
    return train_step

def create_copy_task_train_step(model: RNN, seq_len: int, padding_token: int):
    """Create a JIT-compiled training step for copy task"""
    
    @jax.jit
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, learning_rate: float = 0.001):
        """Single training step with gradient descent for copy task"""

        def loss_fn(params):
            logits, _ = model.forward_sequence_copy_task(params, x_batch)
            return copy_task_loss(logits, y_batch, seq_len, padding_token)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Gradient clipping for stability
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)

        # Update parameters
        params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

        return params, loss
    
    return train_step

# Example usage
if __name__ == '__main__':
    # Workaround for JAX Metal issues - try CPU backend first
    try:
        jax.config.update('jax_platform_name', 'cpu')
        print("Using CPU backend")
    except:
        print("Using default backend")
    
    # create model
    try:
        key = random.PRNGKey(42)
    except Exception as e:
        print(f"Error creating random key: {e}")
        print("Try running with: JAX_PLATFORM_NAME=cpu python rnn.py")
        exit(1)
    # Example configuration for character-level modeling
    vocab_size = 50  # Example vocabulary size
    embedding_dim = 128
    hidden_size = 256
    
    model = create_rnn_model(vocab_size=vocab_size, embedding_dim=embedding_dim, 
                           hidden_size=hidden_size, output_size=vocab_size,
                           num_layers=2, cell_type='lstm')  # Change to 'rnn' for basic RNN
    
    # Initialize parameters
    params = model.init_params(key)

    # Create dummy character index data
    batch_size, seq_len = 32, 20
    x = random.randint(key, (batch_size, seq_len), 0, vocab_size)  # Character indices
    y = random.randint(key, (batch_size, seq_len), 0, vocab_size)  # Target character indices

    # Training loop for multiple epochs
    num_epochs = 15
    learning_rate = 0.001

    # Test forward step
    outputs, final_states = model.forward_sequence(params, x)
    
    # Create JIT-compiled train step for this model
    train_step = create_train_step(model)

    for epoch in range(num_epochs):
        params, loss = train_step(params, x, y, learning_rate)

        if epoch % 5 == 0: 
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    print(f"Final loss after {num_epochs} epochs: {loss:.4f}")
    
    # Test prediction
    test_logits, _ = model.forward_sequence(params, x[:1])  # Single sequence
    predicted_chars = jnp.argmax(test_logits[0], axis=-1)
    print(f"Input chars: {x[0][:10]}")
    print(f"Predicted chars: {predicted_chars[:10]}")