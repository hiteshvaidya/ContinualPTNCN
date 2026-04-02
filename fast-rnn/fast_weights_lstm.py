"""
Fast Weights Implementation for LSTMs
Based on: "Using Fast Weights to Attend to the Recent Past" (Ba et al., 2016)
https://arxiv.org/pdf/1610.06258

This implementation includes:
- Standard LSTM baseline
- LSTM with Fast Weights mechanism
- Support for text8 and Penn Treebank datasets
- Character-level language modeling
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


class StandardLSTM(nn.Module):
    """Standard LSTM for character-level language modeling - Baseline Model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.0):
        super(StandardLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
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
            elif 'output_projection' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x, hidden_states=None):
        """
        Standard LSTM forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden_states: Tuple of (h, c) hidden states
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Updated (h, c) hidden states
        """
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden_states = self.lstm(embedded, hidden_states)
        # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden_states


class FastWeightLSTM(nn.Module):
    """LSTM with Fast Weights mechanism
    
    Implements fast associative memory on top of LSTM hidden states
    A(t) = λA(t-1) + η*h(t)h(t)^T
    For s in 1..S: h_s = f(h_s^{s-1} + η * A(t) * h_s^{s-1})
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1,
                 use_layer_norm=True, lambda_decay=0.95, eta_lr=0.5, S=1, dropout=0.0):
        super(FastWeightLSTM, self).__init__()
        
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for fast weight updates
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
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
            elif 'output_projection' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x, hidden_states=None):
        """
        Forward pass with fast weights applied to LSTM hidden states
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden_states: Tuple of (h, c) hidden states
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Updated (h, c) hidden states
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden_states = self.lstm(embedded, hidden_states)
        # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Apply fast weights to LSTM outputs
        if self.S > 0:
            lstm_out = self._apply_fast_weights(lstm_out)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden_states
    
    def _apply_fast_weights(self, lstm_out):
        """
        Apply fast weights mechanism to LSTM hidden states
        
        Args:
            lstm_out: LSTM outputs (batch_size, seq_len, hidden_size)
        
        Returns:
            enhanced_out: Enhanced outputs with fast weights
        """
        batch_size, seq_len, hidden_size = lstm_out.shape
        enhanced_outputs = []
        
        for t in range(seq_len):
            h_t = lstm_out[:, t, :]  # (batch_size, hidden_size)
            
            if t == 0:
                # No history for first timestep
                enhanced_outputs.append(h_t)
            else:
                # Apply fast weights using previous hidden states
                h_s = h_t
                
                for s in range(self.S):
                    # Get previous hidden states for context
                    # Use all previous timesteps in this sequence
                    prev_states = lstm_out[:, :t, :]  # (batch_size, t, hidden_size)
                    
                    # Compute attention scores
                    attention_scores = torch.bmm(
                        prev_states,  # (batch_size, t, hidden_size)
                        h_s.unsqueeze(2)  # (batch_size, hidden_size, 1)
                    ).squeeze(2)  # (batch_size, t)
                    
                    # Apply temporal decay
                    tau = torch.arange(t, device=lstm_out.device, dtype=torch.float32)
                    decay_weights = self.lambda_decay ** (t - tau)  # (t,)
                    
                    # Weighted attention
                    weighted_attention = attention_scores * decay_weights.unsqueeze(0)  # (batch_size, t)
                    
                    # Normalize
                    attention_weights = F.softmax(weighted_attention, dim=1)  # (batch_size, t)
                    
                    # Compute context vector
                    context = torch.bmm(
                        attention_weights.unsqueeze(1),  # (batch_size, 1, t)
                        prev_states  # (batch_size, t, hidden_size)
                    ).squeeze(1)  # (batch_size, hidden_size)
                    
                    # Fast update
                    fast_update = self.eta_lr * context
                    
                    # Apply layer normalization
                    if self.use_layer_norm:
                        fast_update = self.layer_norm(fast_update)
                    
                    h_s = h_s + fast_update
                
                enhanced_outputs.append(h_s)
        
        # Stack enhanced outputs
        enhanced_out = torch.stack(enhanced_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return enhanced_out


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
        return (len(self.data) - 1) // self.seq_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
        
        chunk = self.data[start_idx:end_idx]
        
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))
        
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
        return (len(self.data) - 1) // self.seq_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
        
        chunk = self.data[start_idx:end_idx]
        
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
        if batch_idx % 100 == 0 or hidden_states is None:
            hidden_states = None
        elif hidden_states[0].size(1) != current_batch_size:
            hidden_states = None
        
        # Forward pass
        logits, hidden_states = model(x, hidden_states)
        
        # Detach hidden states to truncate backprop
        if hidden_states is not None:
            hidden_states = tuple(h.detach() for h in hidden_states)
        
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
            if hidden_states is not None and hidden_states[0].size(1) != current_batch_size:
                hidden_states = None
            
            logits, hidden_states = model(x, hidden_states)
            
            # Detach hidden states
            if hidden_states is not None:
                hidden_states = tuple(h.detach() for h in hidden_states)
            
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
    parser = argparse.ArgumentParser(description='Fast Weights LSTM Training')
    parser.add_argument('--dataset', type=str, default='text8', choices=['text8', 'ptb'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='data/text8',
                       help='Path to dataset')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'fast_lstm'],
                       help='Model architecture')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (only for num_layers > 1)')
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
    parser.add_argument('--save_path', type=str, default='checkpoints/fast_weights_lstm.pt',
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
    if args.model == 'lstm':
        model = StandardLSTM(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # fast_lstm
        model = FastWeightLSTM(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            use_layer_norm=args.use_layer_norm,
            lambda_decay=args.lambda_decay,
            eta_lr=args.eta_lr,
            S=args.S,
            dropout=args.dropout
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
