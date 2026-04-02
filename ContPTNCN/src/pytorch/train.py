"""
Training script for EmbeddingPTNCN on WikiText-2.

No backprop — weight updates come entirely from compute_updates() (LRA rule).
Cross-entropy loss is computed only for monitoring (perplexity).
"""

import math
import torch
import torch.nn.functional as F

from data_module import WikiTextLoader, load_wikitext2, load_ptb
from network import EmbeddingPTNCN

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATASET     = "ptb"   # "wikitext2" or "ptb"
VOCAB_SIZE  = None          # set automatically from vocab after loading
EMB_DIM     = 64
HID_DIM     = 256

BATCH_SIZE  = 32
SEQ_LEN     = 35        # number of timesteps per chunk (like standard LM)
N_EPOCHS    = 5

ALPHA       = 0.001 / BATCH_SIZE   # LRA learning rate (per-sample scale)
XI          = 0.4       # Hebbian regularisation strength
BETA        = 0.1       # LRA target shift
GAMMA       = 0.01
LAMBDA_VAL  = 0.01
CLIP_NORM   = 1.0       # max Frobenius norm per LRA update matrix; None to disable

LOG_INTERVAL = 200      # print every N batches
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model: EmbeddingPTNCN, loader: WikiTextLoader, device: torch.device):
    """Return (perplexity, accuracy) on a data split (no weight updates)."""
    model._clear_state()
    total_loss    = 0.0
    total_correct = 0
    n_tokens      = 0

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
                preds = logits.argmax(dim=-1)             # (B,)
                total_correct += (preds == labels[:, t]).sum().item()
                total_loss    += loss.item() * tokens.shape[0]
                n_tokens      += tokens.shape[0]

    avg_loss = total_loss / n_tokens
    ppl = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)
    acc = total_correct / n_tokens
    return ppl, bpc, acc


def train():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders = {"wikitext2": load_wikitext2, "ptb": load_ptb}
    if DATASET not in loaders:
        raise ValueError(f"Unknown dataset '{DATASET}'. Choose from: {list(loaders)}")
    print(f"Loading {DATASET} …")
    train_ids, val_ids, test_ids, vocab = loaders[DATASET]()
    vocab_size = len(vocab)
    print(f"  train tokens: {len(train_ids):,}  |  val: {len(val_ids):,}  |  test: {len(test_ids):,}")
    print(f"  vocab size:   {vocab_size:,}")

    train_loader = WikiTextLoader(train_ids, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=False)
    val_loader   = WikiTextLoader(val_ids,   batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    test_loader  = WikiTextLoader(test_ids,  batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EmbeddingPTNCN(vocab_size=vocab_size, emb_dim=EMB_DIM, hid_dim=HID_DIM)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, N_EPOCHS + 1):
        model._clear_state()        # reset temporal state at epoch start
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_tokens  = 0
        running_loss    = 0.0
        running_correct = 0
        running_tokens  = 0

        for batch_idx, batch in enumerate(train_loader):
            tokens = batch.tokens.to(device)   # (B, T)
            labels = batch.labels.to(device)   # (B, T)
            mask   = batch.mask.to(device)     # (B, 1)
            T = tokens.shape[1]

            for t in range(T):
                # PTNCN never calls .backward() — no_grad prevents graph build-up
                with torch.no_grad():
                    logits = model(tokens[:, t], mask,
                                   beta=BETA, gamma=GAMMA, lambda_val=LAMBDA_VAL)
                    loss  = F.cross_entropy(logits, labels[:, t])
                    preds = logits.argmax(dim=-1)           # (B,)
                    correct = (preds == labels[:, t]).sum().item()

                model.compute_updates(alpha=ALPHA, xi=XI, max_norm=CLIP_NORM)

                B_t = tokens.shape[0]
                running_loss    += loss.item()
                running_correct += correct
                running_tokens  += B_t
                epoch_loss      += loss.item() * B_t
                epoch_correct   += correct
                epoch_tokens    += B_t

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = running_loss / (LOG_INTERVAL * T)
                avg_acc  = running_correct / running_tokens
                ppl      = math.exp(avg_loss)
                bpc      = avg_loss / math.log(2)
                print(f"  epoch {epoch} | batch {batch_idx+1:>5d} | "
                      f"loss {avg_loss:.4f} | ppl {ppl:>8.2f} | bpc {bpc:.4f} | acc {avg_acc:.4f}")
                running_loss = running_correct = running_tokens = 0

        train_loss = epoch_loss / epoch_tokens
        train_ppl = math.exp(train_loss)
        train_bpc = train_loss / math.log(2)
        train_acc = epoch_correct / epoch_tokens
        val_ppl, val_bpc, val_acc = evaluate(model, val_loader, device)
        model._clear_state()        # restore state after eval

        stats = model.weight_stats()
        w_norms = " ".join(f"{k.replace('ptncn.','').replace('_norm','')}={v:.2f}"
                           for k, v in stats.items() if k.endswith("_norm"))
        w_maxes = " ".join(f"{k.replace('ptncn.','').replace('_max','')}={v:.3f}"
                           for k, v in stats.items() if k.endswith("_max"))
        print(f"Epoch {epoch} done | "
              f"train ppl {train_ppl:.2f} bpc {train_bpc:.4f} acc {train_acc:.4f} | "
              f"val ppl {val_ppl:.2f} bpc {val_bpc:.4f} acc {val_acc:.4f}")
        print(f"  weight norms : {w_norms}")
        print(f"  weight maxabs: {w_maxes}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_ppl, test_bpc, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest perplexity: {test_ppl:.2f} | bpc: {test_bpc:.4f} | accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train()
