"""
Fast Weights Implementation for RNNs
Based on: "Using Fast Weights to Attend to the Recent Past" (Ba et al., 2016)
https://arxiv.org/pdf/1610.06258

This implementation includes:
- Fast associative memory using outer product updates
- Layer normalization for stability
- Support for text8 and Penn Treebank datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import zipfile
import urllib.request
from tqdm import tqdm
import argparse
import math


class StandardRNN(nn.Module):
    """Standard RNN without Fast Weights - Baseline Model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(StandardRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(embedding_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for all layers"""
        return [torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)]
    
    def forward(self, x, hidden_states=None):
        """
        Standard RNN forward pass without fast weights
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden_states: List of hidden states for each layer
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Updated hidden states
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Store outputs for each timestep
        outputs = []
        
        # Process sequence timestep by timestep
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                h_prev = hidden_states[layer_idx]
                
                # Standard RNN update (no fast weights)
                h_new = self.rnn_cells[layer_idx](x_t, h_prev)  # (batch_size, hidden_size)
                
                # Update hidden state
                hidden_states[layer_idx] = h_new
                
                # Use this layer's output as next layer's input
                x_t = h_new
            
            # Store output from last layer
            outputs.append(h_new)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        # Project to vocabulary
        logits = self.output_projection(outputs)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden_states


class FastWeightRNN(nn.Module):
    """RNN with Fast Weights mechanism
    
    Implements the fast associative memory from Ba et al. 2016
    A(t) = λA(t-1) + η*h(t)h(t)^T
    For s in 1..S: h_s = f(W*x + C*h_{s-1} + A(t)*h_{s-1})
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1,
                 use_layer_norm=True, lambda_decay=0.95, eta_lr=0.5, S=1):
        super(FastWeightRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Fast weight hyperparameters
        self.lambda_decay = lambda_decay  # λ: decay rate for fast weights
        self.eta_lr = eta_lr  # η: learning rate for fast weights
        self.S = S  # Number of inner loop iterations
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(embedding_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Layer normalization for fast weight updates
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for all layers"""
        return [torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)]
    
    def forward(self, x, hidden_states=None):
        """
        Forward pass with fast weights
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden_states: List of hidden states for each layer
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Updated hidden states
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Store outputs for each timestep
        outputs = []
        
        # Process sequence timestep by timestep
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                h_prev = hidden_states[layer_idx]
                
                # Standard RNN update
                h_new = self.rnn_cells[layer_idx](x_t, h_prev)  # (batch_size, hidden_size)
                
                # Apply fast weights mechanism
                if self.S > 0 and t > 0:
                    h_new = self._apply_fast_weights(
                        h_new, 
                        hidden_states[layer_idx],
                        layer_idx,
                        t
                    )
                
                # Update hidden state
                hidden_states[layer_idx] = h_new
                
                # Use this layer's output as next layer's input
                x_t = h_new
            
            # Store output from last layer
            outputs.append(h_new)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        # Project to vocabulary
        logits = self.output_projection(outputs)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden_states
    
    def _apply_fast_weights(self, h_t, h_prev, layer_idx, t):
        """
        Apply fast weights mechanism
        
        h_s^0 = h_t (from slow weights)
        For s in 1..S:
            h_s = f(h_s^{s-1} + η * A(t) * h_s^{s-1})
        
        Where A(t) is approximated by attention over recent hidden states
        """
        h_s = h_t
        
        for s in range(self.S):
            # Fast weight update: η * A(t) * h_s
            # We approximate A(t) * h_s using dot product attention
            
            # In practice, we compute: sum_tau (lambda^(t-tau) * h_tau * dot(h_tau, h_s))
            # This is equivalent to A(t) * h_s where A accumulates outer products
            
            # For simplicity and efficiency, we use the current and previous hidden state
            # Fast update = η * h_prev * dot(h_prev, h_s) * lambda
            
            dot_product = torch.sum(h_prev * h_s, dim=-1, keepdim=True)  # (batch_size, 1)
            fast_update = self.eta_lr * h_prev * dot_product  # (batch_size, hidden_size)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                fast_update = self.layer_norms[layer_idx](fast_update)
            
            # Update h_s
            h_s = h_s + fast_update
        
        return h_s


class FastWeightRNNWithHistory(nn.Module):
    """
    Fast Weight RNN that maintains full history for more accurate implementation
    This follows the paper more closely by maintaining A(t) through all timesteps
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1,
                 use_layer_norm=True, lambda_decay=0.95, eta_lr=0.5, S=1):
        super(FastWeightRNNWithHistory, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Fast weight hyperparameters
        self.lambda_decay = lambda_decay
        self.eta_lr = eta_lr
        self.S = S
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(embedding_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states"""
        return [torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)]
    
    def forward(self, x, hidden_states=None):
        """Forward pass with full history-based fast weights"""
        batch_size, seq_len = x.shape
        device = x.device
        
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        # Embed input
        embedded = self.embedding(x)
        
        # Store hidden state history for each layer
        # History shape: (num_layers, batch_size, seq_len, hidden_size)
        history = [[] for _ in range(self.num_layers)]
        
        outputs = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            for layer_idx in range(self.num_layers):
                h_prev = hidden_states[layer_idx]
                
                # Standard RNN update
                h_t = self.rnn_cells[layer_idx](x_t, h_prev)
                
                # Apply fast weights using history
                if self.S > 0 and len(history[layer_idx]) > 0:
                    h_t = self._apply_fast_weights_with_history(
                        h_t,
                        history[layer_idx],
                        layer_idx,
                        t
                    )
                
                # Store in history
                history[layer_idx].append(h_t.detach())  # Detach to save memory
                
                # Update hidden state
                hidden_states[layer_idx] = h_t
                x_t = h_t
            
            outputs.append(h_t)
        
        outputs = torch.stack(outputs, dim=1)
        logits = self.output_projection(outputs)
        
        return logits, hidden_states
    
    def _apply_fast_weights_with_history(self, h_t, history, layer_idx, t):
        """
        Apply fast weights using full history
        
        A(t) ≈ sum_{tau=1}^{t} lambda^(t-tau) * h_tau * h_tau^T
        Fast update = A(t) * h_s
        """
        h_s = h_t
        
        # Convert history to tensor
        h_history = torch.stack(history, dim=1)  # (batch_size, t, hidden_size)
        
        for s in range(self.S):
            # Compute attention scores: h_history * h_s
            attention_scores = torch.bmm(
                h_history,  # (batch_size, t, hidden_size)
                h_s.unsqueeze(2)  # (batch_size, hidden_size, 1)
            ).squeeze(2)  # (batch_size, t)
            
            # Apply temporal decay
            tau = torch.arange(len(history), device=h_s.device, dtype=torch.float32)
            decay_weights = self.lambda_decay ** (t - tau)  # (t,)
            
            # Weighted attention
            weighted_attention = attention_scores * decay_weights.unsqueeze(0)  # (batch_size, t)
            
            # Normalize
            attention_weights = F.softmax(weighted_attention, dim=1)  # (batch_size, t)
            
            # Compute context vector
            context = torch.bmm(
                attention_weights.unsqueeze(1),  # (batch_size, 1, t)
                h_history  # (batch_size, t, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
            
            # Fast update
            fast_update = self.eta_lr * context
            
            # Apply layer normalization
            if self.use_layer_norm:
                fast_update = self.layer_norms[layer_idx](fast_update)
            
            h_s = h_s + fast_update
        
        return h_s


class Text8Dataset(Dataset):
    """Text8 character-level dataset"""
    
    def __init__(self, data_path, seq_length=100, split='train', download=True):
        self.seq_length = seq_length
        self.split = split
        
        # Download if necessary
        if download and not os.path.exists(data_path):
            self._download_text8(data_path)
        
        # Load data
        with open(data_path, 'r') as f:
            data = f.read()
        
        # Create character vocabulary
        chars = sorted(list(set(data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Split data
        data_len = len(data)
        if split == 'train':
            data = data[:int(0.9 * data_len)]
        elif split == 'valid':
            data = data[int(0.9 * data_len):int(0.95 * data_len)]
        else:  # test
            data = data[int(0.95 * data_len):]
        
        # Convert to indices
        self.data = [self.char_to_idx[ch] for ch in data]
    
    def _download_text8(self, data_path):
        """Download text8 dataset"""
        url = 'http://mattmahoney.net/dc/text8.zip'
        print(f"Downloading text8 dataset from {url}")
        
        os.makedirs(os.path.dirname(data_path) or '.', exist_ok=True)
        zip_path = data_path + '.zip'
        
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(data_path) or '.')
        
        os.remove(zip_path)
        print("Download complete!")
    
    def __len__(self):
        # Only return complete sequences
        return (len(self.data) - 1) // self.seq_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        # Ensure we don't go past the end
        if end_idx > len(self.data):
            end_idx = len(self.data)
        
        chunk = self.data[start_idx:end_idx]
        
        # Pad if necessary to ensure consistent size
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))
        
        # Input and target (shifted by 1)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class PTBDataset(Dataset):
    """Penn Treebank character-level dataset"""
    
    def __init__(self, data_dir, seq_length=100, split='train'):
        self.seq_length = seq_length
        self.split = split
        
        # Load data
        if split == 'train':
            file_path = os.path.join(data_dir, 'ptb.train.txt')
        elif split == 'valid':
            file_path = os.path.join(data_dir, 'ptb.valid.txt')
        else:  # test
            file_path = os.path.join(data_dir, 'ptb.test.txt')
        
        with open(file_path, 'r') as f:
            data = f.read()
        
        # Create character vocabulary
        chars = sorted(list(set(data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Convert to indices
        self.data = [self.char_to_idx[ch] for ch in data]
    
    def __len__(self):
        # Only return complete sequences
        return (len(self.data) - 1) // self.seq_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        # Ensure we don't go past the end
        if end_idx > len(self.data):
            end_idx = len(self.data)
        
        chunk = self.data[start_idx:end_idx]
        
        # Pad if necessary to ensure consistent size
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def train_epoch(model, dataloader, optimizer, device, clip_grad=5.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    hidden_states = None
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        
        # Reset hidden states periodically or if batch size changes
        current_batch_size = x.size(0)
        if batch_idx % 100 == 0 or hidden_states is None or hidden_states[0].size(0) != current_batch_size:
            hidden_states = None
        
        # Forward pass
        logits, hidden_states = model(x, hidden_states)
        
        # Detach hidden states to truncate backprop
        if hidden_states is not None:
            hidden_states = [h.detach() for h in hidden_states]
        
        # Compute loss (cross-entropy in nats)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Update statistics (convert to BPC)
        batch_size, seq_len = x.shape
        bpc = loss.item() / math.log(2)
        total_loss += bpc * batch_size * seq_len
        total_tokens += batch_size * seq_len
        
        # Update progress bar
        progress_bar.set_postfix({
            'bpc': f'{bpc:.4f}'
        })
    
    avg_bpc = total_loss / total_tokens
    return avg_bpc


def evaluate(model, dataloader, device):
    """Evaluate model and return BPC"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    hidden_states = None
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Evaluating")):
            x, y = x.to(device), y.to(device)
            
            # Reset hidden states if batch size changes
            current_batch_size = x.size(0)
            if hidden_states is not None and hidden_states[0].size(0) != current_batch_size:
                hidden_states = None
            
            logits, hidden_states = model(x, hidden_states)
            
            # Detach hidden states
            if hidden_states is not None:
                hidden_states = [h.detach() for h in hidden_states]
            
            # Compute loss (cross-entropy in nats)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Convert to BPC
            batch_size, seq_len = x.shape
            bpc = loss.item() / math.log(2)
            total_loss += bpc * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    avg_bpc = total_loss / total_tokens
    return avg_bpc


def generate_text(model, dataset, seed_text="The ", length=200, temperature=1.0, device='cuda'):
    """Generate text from the model"""
    model.eval()
    
    # Encode seed text
    input_ids = [dataset.char_to_idx.get(c, 0) for c in seed_text]
    generated = input_ids.copy()
    
    hidden_states = None
    
    with torch.no_grad():
        # Process seed text
        if len(input_ids) > 0:
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits, hidden_states = model(x, hidden_states)
        
        # Generate new characters
        for _ in range(length):
            # Get last character
            if len(generated) > 0:
                x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            else:
                x = torch.tensor([[0]], dtype=torch.long, device=device)
            
            logits, hidden_states = model(x, hidden_states)
            
            # Sample from distribution
            probs = F.softmax(logits[0, -1] / temperature, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            generated.append(next_char_idx)
    
    # Decode
    generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated])
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Fast Weights RNN Training')
    parser.add_argument('--dataset', type=str, default='text8', choices=['text8', 'ptb'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='data/text8',
                       help='Path to dataset')
    parser.add_argument('--model', type=str, default='fast_rnn', 
                       choices=['standard_rnn', 'fast_rnn', 'fast_rnn_history'],
                       help='Model architecture')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of RNN layers')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda_decay', type=float, default=0.95,
                       help='Fast weight decay rate')
    parser.add_argument('--eta_lr', type=float, default=0.5,
                       help='Fast weight learning rate')
    parser.add_argument('--S', type=int, default=1,
                       help='Number of fast weight inner loop iterations')
    parser.add_argument('--use_layer_norm', action='store_true', default=True,
                       help='Use layer normalization')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--save_path', type=str, default='checkpoints/fast_weights_model.pt',
                       help='Path to save model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'text8':
        train_dataset = Text8Dataset(args.data_path, args.seq_length, 'train')
        valid_dataset = Text8Dataset(args.data_path, args.seq_length, 'valid')
        test_dataset = Text8Dataset(args.data_path, args.seq_length, 'test')
    else:  # ptb
        train_dataset = PTBDataset(args.data_path, args.seq_length, 'train')
        valid_dataset = PTBDataset(args.data_path, args.seq_length, 'valid')
        test_dataset = PTBDataset(args.data_path, args.seq_length, 'test')
    
    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'standard_rnn':
        model = StandardRNN(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
    elif args.model == 'fast_rnn':
        model = FastWeightRNN(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            use_layer_norm=args.use_layer_norm,
            lambda_decay=args.lambda_decay,
            eta_lr=args.eta_lr,
            S=args.S
        )
    else:  # fast_rnn_history
        model = FastWeightRNNWithHistory(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            use_layer_norm=args.use_layer_norm,
            lambda_decay=args.lambda_decay,
            eta_lr=args.eta_lr,
            S=args.S
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_valid_bpc = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_bpc = train_epoch(model, train_loader, optimizer, device, args.clip_grad)
        
        # Validate
        valid_bpc = evaluate(model, valid_loader, device)
        
        print(f"Train BPC: {train_bpc:.4f}")
        print(f"Valid BPC: {valid_bpc:.4f}")
        
        # Save best model
        if valid_bpc < best_valid_bpc:
            best_valid_bpc = valid_bpc
            os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_bpc': valid_bpc,
                'args': args
            }, args.save_path)
            print(f"Saved best model with valid BPC: {valid_bpc:.4f}")
        
        # Generate sample text
        if args.dataset == 'text8':
            sample = generate_text(model, train_dataset, "the ", length=200, device=device)
            print(f"\nGenerated sample:\n{sample}\n")
    
    # Test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    
    test_bpc = evaluate(model, test_loader, device)
    print(f"Test BPC: {test_bpc:.4f}")
    
    # Generate final samples
    if args.dataset == 'text8':
        for seed in ["the ", "in ", "of "]:
            sample = generate_text(model, train_dataset, seed, length=300, device=device)
            print(f"\nGenerated from '{seed}':\n{sample}\n")


if __name__ == '__main__':
    main()
