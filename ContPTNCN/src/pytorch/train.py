"""
Training script for EmbeddingPTNCN on WikiText-2.

No backprop — weight updates come entirely from compute_updates() (LRA rule).
Cross-entropy loss is computed only for monitoring (perplexity).
"""

import math
import torch
import torch.nn.functional as F

from data_module import WikiTextLoader, load_wikitext2
from network import EmbeddingPTNCN

# ── Hyperparameters ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 10_000    # must match max_tokens in build_vocab
EMB_DIM     = 64
HID_DIM     = 256

BATCH_SIZE  = 32
SEQ_LEN     = 35        # number of timesteps per chunk (like standard LM)
N_EPOCHS    = 5

ALPHA       = 0.001     # LRA learning rate
XI          = 0.4       # Hebbian regularisation strength
BETA        = 0.1       # LRA target shift
GAMMA       = 0.01
LAMBDA_VAL  = 0.01

LOG_INTERVAL = 200      # print every N batches
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model: EmbeddingPTNCN, loader: WikiTextLoader, device: torch.device) -> float:
    """Return perplexity on a data split (no weight updates)."""
    model._clear_state()
    total_loss = 0.0
    n_tokens   = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch.tokens.to(device)   # (B, T)
            labels = batch.labels.to(device)   # (B, T)
            mask   = batch.mask.to(device)     # (B, 1)
            T = tokens.shape[1]

            for t in range(T):
                logits = model(tokens[:, t], mask,
                               beta=BETA, gamma=GAMMA, lambda_val=LAMBDA_VAL)
                loss = F.cross_entropy(logits, labels[:, t])
                total_loss += loss.item() * tokens.shape[0]
                n_tokens   += tokens.shape[0]

    return math.exp(total_loss / n_tokens)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading WikiText-2 …")
    train_ids, val_ids, test_ids, vocab = load_wikitext2()
    print(f"  train tokens: {len(train_ids):,}  |  val: {len(val_ids):,}  |  test: {len(test_ids):,}")
    print(f"  vocab size:   {len(vocab):,}")

    train_loader = WikiTextLoader(train_ids, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=False)
    val_loader   = WikiTextLoader(val_ids,   batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    test_loader  = WikiTextLoader(test_ids,  batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmbeddingPTNCN(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hid_dim=HID_DIM)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, N_EPOCHS + 1):
        model._clear_state()        # reset temporal state at epoch start
        epoch_loss = 0.0
        epoch_tokens = 0
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            tokens = batch.tokens.to(device)   # (B, T)
            labels = batch.labels.to(device)   # (B, T)
            mask   = batch.mask.to(device)     # (B, 1)
            T = tokens.shape[1]

            for t in range(T):
                logits = model(tokens[:, t], mask,
                               beta=BETA, gamma=GAMMA, lambda_val=LAMBDA_VAL)

                # cross-entropy for monitoring only — no .backward()
                with torch.no_grad():
                    loss = F.cross_entropy(logits, labels[:, t])

                model.compute_updates(alpha=ALPHA, xi=XI)

                running_loss += loss.item()
                epoch_loss   += loss.item() * tokens.shape[0]
                epoch_tokens += tokens.shape[0]

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg = running_loss / (LOG_INTERVAL * T)
                ppl = math.exp(avg)
                print(f"  epoch {epoch} | batch {batch_idx+1:>5d} | "
                      f"loss {avg:.4f} | ppl {ppl:>8.2f}")
                running_loss = 0.0

        train_ppl = math.exp(epoch_loss / epoch_tokens)
        val_ppl   = evaluate(model, val_loader, device)
        model._clear_state()        # restore state after eval
        print(f"Epoch {epoch} done | train ppl {train_ppl:.2f} | val ppl {val_ppl:.2f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_ppl = evaluate(model, test_loader, device)
    print(f"\nTest perplexity: {test_ppl:.2f}")


if __name__ == "__main__":
    train()
