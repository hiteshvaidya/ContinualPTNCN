# Fast Weights RNN Implementation

PyTorch implementation of "Using Fast Weights to Attend to the Recent Past" (Ba et al., 2016)
https://arxiv.org/pdf/1610.06258

## Features

- ✅ Complete implementation of Fast Weights mechanism for RNNs
- ✅ Three model variants:
  - `StandardRNN`: Baseline RNN without fast weights
  - `FastWeightRNN`: Efficient approximation using previous hidden state
  - `FastWeightRNNWithHistory`: Full implementation with complete history
- ✅ Layer normalization for stability
- ✅ Support for text8 and Penn Treebank datasets
- ✅ Character-level language modeling
- ✅ Gradient clipping and other training stability features

## Algorithm

The Fast Weights mechanism implements an associative memory:

```
A(t) = λ * A(t-1) + η * h(t) ⊗ h(t)^T

For s in 1..S:
    h_s = f(h_s^{s-1} + η * A(t) * h_s^{s-1})
```

Where:
- `λ` (lambda_decay): Decay rate for fast weights (default: 0.95)
- `η` (eta_lr): Learning rate for fast weights (default: 0.5)
- `S`: Number of inner loop iterations (default: 1)
- `A(t)`: Fast associative memory matrix
- `h(t)`: Hidden state at time t

## Installation

```bash
# Install dependencies
pip install torch numpy tqdm

# Or using the project's virtual environment
source /data/hvaidya/ContinualPTNCN/.venv/bin/activate
```

## Usage

### Text8 Dataset

```bash
# Train baseline RNN (no fast weights)
python test/fast_weights.py --dataset text8 --data_path data/text8 --model standard_rnn

# Train with default settings (Fast Weights)
python test/fast_weights.py --dataset text8 --data_path data/text8

# Train with custom hyperparameters
python test/fast_weights.py \
    --dataset text8 \
    --data_path data/text8 \
    --model fast_rnn \
    --hidden_size 512 \
    --embedding_dim 256 \
    --num_layers 2 \
    --S 3 \
    --lambda_decay 0.95 \
    --eta_lr 0.5 \
    --batch_size 64 \
    --epochs 20 \
    --lr 0.0001

# Use full history model
python test/fast_weights.py \
    --dataset text8 \
    --model fast_rnn_history \
    --S 2
```

### Penn Treebank Dataset

```bash
# Assuming PTB data is in ContPTNCN/data/ptb_char/
python test/fast_weights.py \
    --dataset ptb \
    --data_path /data/hvaidya/ContinualPTNCN/ContPTNCN/data/ptb_char \
    --hidden_size 256 \
    --batch_size 32 \
    --epochs 50
```

## Command Line Arguments

### Model Architecture
- `--model`: Model variant (`standard_rnn`, `fast_rnn`, or `fast_rnn_history`)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_size`: Hidden state size (default: 256)
- `--num_layers`: Number of RNN layers (default: 1)

### Fast Weights Parameters
- `--lambda_decay`: Decay rate λ for fast weights (default: 0.95)
- `--eta_lr`: Learning rate η for fast weights (default: 0.5)
- `--S`: Number of inner loop iterations (default: 1)
- `--use_layer_norm`: Use layer normalization (default: True)

### Training
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--seq_length`: Sequence length (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--clip_grad`: Gradient clipping threshold (default: 5.0)

### Data
- `--dataset`: Dataset to use (`text8` or `ptb`)
- `--data_path`: Path to dataset
- `--save_path`: Path to save model checkpoint

## Expected Results

### Text8 Dataset
- Baseline RNN (S=0): ~1.6-1.7 BPC
- Fast Weights (S=1): ~1.5-1.6 BPC
- Fast Weights (S=3): ~1.4-1.5 BPC

### Penn Treebank
- Baseline RNN (S=0): ~1.4-1.5 BPC
- Fast Weights (S=1): ~1.3-1.4 BPC
- Fast Weights (S=3): ~1.2-1.3 BPC

## Model Comparison

### StandardRNN (Baseline)
- No fast weights mechanism
- Standard RNN forward pass only
- Lowest memory usage
- Fastest training
- Good baseline for comparison

### FastWeightRNN (Efficient)
- Memory efficient
- Uses only previous hidden state
- Faster training
- Good for long sequences

### FastWeightRNNWithHistory (Accurate)
- Maintains full hidden state history
- More faithful to paper's algorithm
- Higher memory usage
- Better performance on shorter sequences

## Example Training Session

```bash
cd /data/hvaidya/ContinualPTNCN

# Train baseline RNN on text8 (for comparison)
python test/fast_weights.py \
    --dataset text8 \
    --data_path data/text8 \
    --model standard_rnn \
    --hidden_size 256 \
    --num_layers 1 \
    --batch_size 64 \
    --seq_length 100 \
    --epochs 20 \
    --lr 0.001 \
    --save_path checkpoints/standard_rnn_text8.pt

# Train Fast Weights RNN on text8
python test/fast_weights.py \
    --dataset text8 \
    --data_path data/text8 \
    --model fast_rnn \
    --hidden_size 256 \
    --num_layers 1 \
    --S 1 \
    --lambda_decay 0.95 \
    --eta_lr 0.5 \
    --batch_size 64 \
    --seq_length 100 \
    --epochs 20 \
    --lr 0.001 \
    --save_path checkpoints/fast_weights_text8.pt
```

Expected output:
```
Using device: cuda
Loading text8 dataset...
Vocabulary size: 27
Creating fast_rnn model...
Number of parameters: 1,234,567

Epoch 1/20
Training: 100%|████████| 1234/1234 [01:23<00:00, 14.8it/s, loss=2.1234, bpc=3.0623]
Evaluating: 100%|████████| 123/123 [00:05<00:00, 23.4it/s]
Train Loss: 2.1234 | Train BPC: 3.0623
Valid Loss: 1.9876 | Valid BPC: 2.8654
Saved best model with valid loss: 1.9876

Generated sample:
the quick brown fox jumps over the lazy dog and runs...
```

## Tips for Best Performance

1. **Start with S=1** and increase gradually
2. **Use layer normalization** for stability
3. **Gradient clipping** is essential (5.0 works well)
4. **Lower learning rate** for fast weights (η=0.1-0.5)
5. **Higher decay rate** for longer context (λ=0.95-0.99)

## Troubleshooting

### NaN Loss
- Reduce `--eta_lr` (try 0.1 or 0.2)
- Reduce `--S` (try 1)
- Increase `--clip_grad` (try 1.0)
- Make sure `--use_layer_norm` is enabled

### Out of Memory
- Reduce `--batch_size`
- Reduce `--seq_length`
- Use `fast_rnn` instead of `fast_rnn_history`
- Reduce `--hidden_size`

### Slow Training
- Reduce `--S` (each inner loop iteration adds overhead)
- Use `fast_rnn` instead of `fast_rnn_history`
- Increase `--batch_size` if memory allows

## References

- Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). Using fast weights to attend to the recent past. In Advances in Neural Information Processing Systems (pp. 4331-4339).
- Paper: https://arxiv.org/pdf/1610.06258
