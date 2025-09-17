import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, List
import os

class PTBDataLoader:
    """Penn Treebank character-level data loader for JAX RNN"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, seq_len: int = 35):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        
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
            # If vocab file doesn't exist, create from train data
            print("Vocab file not found, creating from train data...")
            train_file = os.path.join(self.data_dir, 'trainX.txt')
            with open(train_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chars = sorted(list(set(text)))
            for idx, char in enumerate(chars):
                char_to_idx[char] = idx
                idx_to_char[idx] = char
            
            # Save vocab file
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                for char in chars:
                    f.write(char + '\n')
        
        return char_to_idx, idx_to_char
    
    def _load_and_encode(self, file_path: str) -> np.ndarray:
        """Load text file and encode as integer sequence"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode text to integers
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Handle unknown characters (shouldn't happen with proper vocab)
                encoded.append(self.char_to_idx.get('<UNK>', 0))
        
        return np.array(encoded, dtype=np.int32)
    
    def _create_batches(self, data: np.ndarray) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create batches of sequences for training"""
        # Calculate number of batches
        total_len = len(data)
        batch_len = total_len // self.batch_size
        
        # Trim data to fit exactly into batches
        data = data[:batch_len * self.batch_size]
        data = data.reshape(self.batch_size, batch_len)
        
        # Create sequences
        num_batches = (batch_len - 1) // self.seq_len
        
        for i in range(0, num_batches * self.seq_len, self.seq_len):
            # Input sequences
            x = data[:, i:i + self.seq_len]
            # Target sequences (shifted by 1)
            y = data[:, i + 1:i + self.seq_len + 1]
            
            # Convert to one-hot for input
            x_onehot = jnp.eye(self.vocab_size)[x]  # (batch_size, seq_len, vocab_size)
            y_indices = jnp.array(y)  # (batch_size, seq_len)
            
            yield x_onehot, y_indices
    
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
