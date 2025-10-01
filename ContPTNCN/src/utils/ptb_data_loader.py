import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, List
import os

class PTBDataLoader:
    """Penn Treebank character-level data loader for JAX RNN"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, 
                 seq_len: int = 35, padding: int=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        
        # Load vocabulary and create mappings
        self.vocab_file = os.path.join(data_dir, 'vocab.txt')
        self.char_to_idx, self.idx_to_char = self._load_vocab()
        self.vocab_size = len(self.char_to_idx)
        
        # Set padding token automatically as the tilde character
        if padding is None:
            self.padding = self.char_to_idx.get('~', self.vocab_size - 1)
        else:
            self.padding = padding
            
        # Ensure padding token is within vocabulary bounds
        if self.padding >= self.vocab_size:
            print(f"Warning: padding token {self.padding} is out of bounds (vocab size: {self.vocab_size})")
            self.padding = self.char_to_idx.get('~', self.vocab_size - 1)
            print(f"Using padding token: {self.padding} ('{self.idx_to_char[self.padding]}')")
        
        # Load datasets
        self.train_data = self._load_and_encode(os.path.join(data_dir, 'trainX.txt'))
        self.valid_data = self._load_and_encode(os.path.join(data_dir, 'validX.txt'))
        self.test_data = self._load_and_encode(os.path.join(data_dir, 'testX.txt'))
        
        print(f"Loaded PTB dataset:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Train data length: {len(self.train_data)}")
        print(f"  Valid data length: {len(self.valid_data)}")
        print(f"  Test data length: {len(self.test_data)}")
        print(f" Sequence length: {self.seq_len}, Batch size: {self.batch_size}")
    
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
    
    def _create_batches(self, data: np.ndarray, task: str) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
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
        
        print(f"task: {task}, total_len: {total_len}, batch_len: {batch_len}, num_batches: {num_batches}")

        if task == 'copy':
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
                                    data[:, i:i + self.seq_len]], 
                                    axis=1)
                
                yield x_indices, y_indices
        elif task == 'next_char':
            for i in range(0, num_batches * self.seq_len, self.seq_len):
                x_indices = jnp.array(data[:, i:i + self.seq_len])
                y_indices = jnp.array(data[:, i + 1:i + self.seq_len + 1])
                yield x_indices, y_indices
    
    def get_train_batches(self, task='copy') -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get training data batches"""
        return self._create_batches(self.train_data, task)
    
    def get_valid_batches(self, task=None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get validation data batches"""
        return self._create_batches(self.valid_data, task)

    def get_test_batches(self, task=None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get test data batches"""
        return self._create_batches(self.test_data, task)

    def decode_sequence(self, indices) -> str:
        """Decode integer sequence back to text"""
        # Convert to numpy array if it's a JAX array
        if hasattr(indices, 'shape'):  # Check if it's an array-like object
            indices = np.array(indices)
        
        # Convert to regular Python integers for dictionary lookup
        return ''.join([self.idx_to_char[int(idx)] for idx in indices if int(idx) in self.idx_to_char])


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
    for batch in data_loader._create_batches(data_loader.train_data, 'next_char'):
        x_indices, y_indices = batch
        for x, y in zip(x_indices, y_indices):
            print(f"x: {data_loader.decode_sequence(x)}")
            print(f"y: {data_loader.decode_sequence(y)}")
            print("-----")
        break  # Just show the first batch

    # for x_indices, y_indices in data_loader.get_train_batches():
    #     print(f"x_indices: {x_indices}")
    #     print(f"y_indices: {y_indices}")
    #     exit(0)
    # text = "theuniversityofsouthflorida"
    # encoded = [data_loader.char_to_idx.get(c, 0) for c in text] + [data_loader.padding] * len(text)
    # print(f"Encoded '{text}' to {encoded}")
    # decoded = data_loader.decode_sequence(encoded)
    # print(f"Decoded {encoded} back to '{decoded}'")