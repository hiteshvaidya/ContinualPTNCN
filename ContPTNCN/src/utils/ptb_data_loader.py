import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, List
import os

class PTBDataLoader:
    """Penn Treebank character-level data loader for JAX RNN"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, 
                 seq_len: int = 35, padding: int=50):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.padding = padding
        
        # Load vocabulary and create mappings
        self.vocab_file = os.path.join(data_dir, 'vocab.txt')
        self.char_to_idx, self.idx_to_char = self._load_vocab()
        self.vocab_size = len(self.char_to_idx)
        
        # Load datasets
        self.train_data = self._load_and_encode(os.path.join(data_dir, 'trainX.txt'))
        self.valid_data = self._load_and_encode(os.path.join(data_dir, 'validX.txt'))
        self.test_data = self._load_and_encode(os.path.join(data_dir, 'testX.txt'))
        
        print(f"Loaded PTB dataset:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Train data length: {len(self.train_data)}")
        print(f"  Valid data length: {len(self.valid_data)}")
        print(f"  Test data length: {len(self.test_data)}")
    
    def _load_vocab(self) -> Tuple[dict, dict]:
        """Load vocabulary from vocab.txt"""
        char_to_idx = {}
        idx_to_char = {}
        
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                chars = f.read().strip().split('\n')
            for idx, char in enumerate(chars):
                char_to_idx[char] = idx
                idx_to_char[idx] = char
        else:
            raise FileNotFoundError(f"Vocabulary file {self.vocab_file} not found")
        
        return char_to_idx, idx_to_char
    
    def _load_and_encode(self, file_path: str) -> np.ndarray:
        """Load pre-tokenized integer file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Parse comma-separated integers
        encoded = []
        for line in content.split('\n'):
            if line.strip():
                # Split by comma and convert to integers
                line_tokens = [int(x.strip()) for x in line.split(',') if x.strip().isdigit()]
                encoded.extend(line_tokens)
        
        return np.array(encoded, dtype=np.int32)
    
    def _create_batches(self, data: np.ndarray) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create batches of sequences for training on copy task
        ex, x: [1,2,3,<pad>,<pad>,<pad>]
            y: [<pad>,<pad>,<pad>,1,2,3]"""
        # Calculate number of batches
        total_len = len(data)
        batch_len = total_len // self.batch_size
        
        # Trim data to fit exactly into batches
        data = data[:batch_len * self.batch_size]
        data = data.reshape(self.batch_size, batch_len)
        
        # Create sequences
        num_batches = (batch_len - 1) // self.seq_len
        # [<pad>...] of size [batch_size, seq_len]
        batchPadding = jnp.full((self.batch_size, self.seq_len), self.padding)
        print(f"data: {data}")
        
        for i in range(0, num_batches * self.seq_len, self.seq_len):
            # Input sequences [character indices, <padding> * seq_len]
            # (batch_size, 2*seq_len)
            x_indices = jnp.concatenate(
                                [jnp.array(data[:, i:i + self.seq_len]), 
                                batchPadding], 
                                axis=1)
            # Target sequences (shifted by seq_len for copy task prediction)
            # (batch_size, 2*seq_len)
            y_indices = jnp.concatenate(
                                [batchPadding, 
                                data[:, i + 1:i + self.seq_len + 1]], 
                                axis=1)
            
            yield x_indices, y_indices
    
    def get_train_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get training data batches"""
        return self._create_batches(self.train_data)
    
    def get_valid_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get validation data batches"""
        return self._create_batches(self.valid_data)
    
    def get_test_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get test data batches"""
        return self._create_batches(self.test_data)
    
    def decode_sequence(self, indices: np.ndarray) -> str:
        """Decode integer sequence back to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])


if __name__ == '__main__':
    # Data loading
    data_dir = "../../data/ptb_char"
    print(f"Loading data from {data_dir}")
    
    try:
        data_loader = PTBDataLoader(data_dir, batch_size=20, seq_len=35)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the PTB data files exist in ../data/ptb_char/")
        exit(1)
    
    print(f"data: {data_loader.train_data[:10]}")
    for x_indices, y_indices in data_loader.get_train_batches():
        print(f"x_indices: {x_indices}")
        print(f"y_indices: {y_indices}")
        exit(0)