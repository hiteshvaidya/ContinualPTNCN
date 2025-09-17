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
        W_hh_orthogonal = self._orthogonal_init(W_hh_init)

        params = {
            'W_ih': random.normal(k1, (self.hidden_size, self.input_size)) * w_ih_std,
            'W_hh': W_hh_orthogonal,
            'b_h': jnp.zeros((self.hidden_size))
        }
        return params
    
    def _orthogonal_init(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Orthogonal initialization using QR decomposition"""
        q, r = jnp.linalg.qr(matrix)
        # Make sure the diagonal of R is positive
        d = jnp.diag(r)
        q = q * jnp.sign(d)
        return q
    
    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        """Initialize hidden state"""
        return jnp.zeros((batch_size, self.hidden_size))
    
    def __call__(self, params: dict, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """forward pass for one time step"""
        h_new = self.activation(
            jnp.dot(x, params['W_ih'].T) +
            jnp.dot(h, params['W_hh'].T) +
            params['b_h']
            )
        return h_new
    
class LSTMCell:
    """LSTM Cell implementation in JAX
    
    Uses Xavier/Glorot initialization for input-to-hidden weights and
    orthogonal initialization for hidden-to-hidden weights for all gates
    to improve training stability and gradient flow.
    """

    def __init__(self, hidden_size: int, input_size: int):
        self.hidden_size = hidden_size
        self.input_size = input_size

    def init_params(self, key: jax.random.PRNGKey) -> dict:
        """Initialize LSTM parameters"""
        keys = random.split(key, 12)

        # Xavier initialization for input-to-hidden weights
        w_ih_std = jnp.sqrt(2.0 / (self.input_size + self.hidden_size))
        
        # Orthogonal initialization for hidden-to-hidden weights
        W_hf_init = random.normal(keys[4], (self.hidden_size, self.hidden_size))
        W_hi_init = random.normal(keys[5], (self.hidden_size, self.hidden_size))
        W_hg_init = random.normal(keys[6], (self.hidden_size, self.hidden_size))
        W_ho_init = random.normal(keys[7], (self.hidden_size, self.hidden_size))
        
        W_hf_orthogonal = self._orthogonal_init(W_hf_init)
        W_hi_orthogonal = self._orthogonal_init(W_hi_init)
        W_hg_orthogonal = self._orthogonal_init(W_hg_init)
        W_ho_orthogonal = self._orthogonal_init(W_ho_init)

        params = {
            # Input-to-hidden weights (forget, input, candidate, output gates)
            'W_if': random.normal(keys[0], (self.hidden_size, self.input_size)) * w_ih_std,
            'W_ii': random.normal(keys[1], (self.hidden_size, self.input_size)) * w_ih_std,
            'W_ig': random.normal(keys[2], (self.hidden_size, self.input_size)) * w_ih_std,
            'W_io': random.normal(keys[3], (self.hidden_size, self.input_size)) * w_ih_std,

            # Hidden-to-hidden weights (orthogonal initialization)
            'W_hf': W_hf_orthogonal,
            'W_hi': W_hi_orthogonal,
            'W_hg': W_hg_orthogonal,
            'W_ho': W_ho_orthogonal,

            # Biases (initialize forget gate bias to 1)
            'b_f': jnp.ones((self.hidden_size,)),
            'b_i': jnp.zeros((self.hidden_size,)),
            'b_g': jnp.zeros((self.hidden_size,)),
            'b_o': jnp.zeros((self.hidden_size,)),
        }
        return params
    
    def _orthogonal_init(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Orthogonal initialization using QR decomposition"""
        q, r = jnp.linalg.qr(matrix)
        # Make sure the diagonal of R is positive
        d = jnp.diag(r)
        q = q * jnp.sign(d)
        return q

    def init_hidden(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize hidden and cell states"""
        h = jnp.zeros((batch_size, self.hidden_size))
        c = jnp.zeros((batch_size, self.hidden_size))
        return (h, c)
    
    def __call__(self, params: dict, x: jnp.ndarray, state: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """forward pass for one time step"""
        h, c = state

        # Forget gate
        f = jax.nn.sigmoid(
            jnp.dot(x, params['W_if'].T) +
            jnp.dot(h, params['W_hf'].T) +
            params['b_f']
        )

        # Input gate
        i = jax.nn.sigmoid(
            jnp.dot(x, params['W_ii'].T) + 
            jnp.dot(h, params['W_hi'].T) + 
            params['b_i']
        )

        # Candidate values
        g = jnp.tanh(
            jnp.dot(x, params['W_ig'].T) +
            jnp.dot(h, params['W_hg'].T) + 
            params['b_g']
        )

        # Output gate
        o = jax.nn.sigmoid(
            jnp.dot(x, params['W_io'].T) +
            jnp.dot(h, params['W_ho'].T) +
            params['b_o']
        )

        # Update cell state
        c_new = f * c + i * g

        # Update hidden state
        h_new = o * jnp.tanh(c_new)

        return h_new, c_new
    
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
            self.cells = [LSTMCell(hidden_size, embedding_dim if i == 0 else hidden_size) for i in range(num_layers)]
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
    
    def init_hidden_states(self, batch_size: int):
        """Initialize hidden states for all layers"""
        # if self.cell_type == 'lstm':
        #     return [cell.init_hidden(batch_size) for cell in self.cells]
        # else:
        return [cell.init_hidden(batch_size) for cell in self.cells]
    
    def forward_step(self, params: dict, x: jnp.ndarray, states):
        """Forward pass for one time step"""
        current_input = x
        new_states = []

        for i, cell in enumerate(self.cells):
            layer_params = params[f'layer_{i}']

            if self.cell_type == 'lstm':
                h_new, c_new = cell(layer_params, current_input, states[i])
                new_states.append((h_new, c_new))
                current_input = h_new
            else:
                h_new = cell(layer_params, current_input, states[i])
                new_states.append(h_new)
                current_input = h_new

        # Output layer
        output = jnp.dot(current_input, params['W_out'].T) + params['b_out']

        return output, new_states
    
    def forward_sequence(self, params: dict, x_seq: jnp.ndarray, initial_states=None):
        """Forward pass for character index sequences"""
        batch_size, seq_len = x_seq.shape  # x_seq contains character indices

        # Convert indices to embeddings
        embedded_seq = params['embedding'][x_seq]  # Shape: (batch_size, seq_len, embedding_dim)

        if initial_states is None:
            states = self.init_hidden_states(batch_size)
        else:
            states = initial_states

        outputs = []

        for t in range(seq_len):
            output, states = self.forward_step(params, embedded_seq[:, t, :], states)
            outputs.append(output)

        return jnp.stack(outputs, axis=1), states  # Shape: (batch_size, seq_len, vocab_size)
    
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
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    
    # Select the log probabilities of the target tokens
    target_log_probs = log_probs[jnp.arange(targets_flat.shape[0]), targets_flat]
    
    # Return negative mean log likelihood
    return -jnp.mean(target_log_probs)

def create_train_step(model: RNN):
    """Create a JIT-compiled training step for character prediction"""
    
    @jax.jit
    def train_step(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray, learning_rate: float = 0.001):
        """Single training step with gradient descent for character prediction"""

        def loss_fn(params):
            logits, _ = model.forward_sequence(params, x_batch)
            return character_prediction_loss(logits, y_batch)
        
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
    print(f"Input shape: {x.shape} (character indices)")
    print(f"Output shape: {outputs.shape} (logits over vocabulary)")

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