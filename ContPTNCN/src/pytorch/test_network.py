import torch
import pytest
from network import PTNCN, EmbeddingPTNCN

# --- helpers ---
VOCAB  = 100
HID    = 64
IN_DIM = 32
EMB    = 16

def make_ptncn(**kwargs):
    defaults = dict(x_dim=VOCAB, hid_dim=HID, in_dim=IN_DIM)
    defaults.update(kwargs)
    return PTNCN(**defaults)

# --- __init__ weight shape tests ---
def test_weight_shapes():
    model = make_ptncn()
    assert model.M1.shape == (IN_DIM, HID)
    assert model.M2.shape == (HID, HID)
    assert model.W1.shape == (HID, IN_DIM)
    assert model.W2.shape == (HID, HID)
    assert model.V1.shape == (HID, HID)
    assert model.V2.shape == (HID, HID)
    assert model.E1.shape == (IN_DIM, HID)
    assert model.E2.shape == (HID, HID)
    assert model.U1.shape == (HID, HID)

def test_u1_absent_when_zeta_zero():
    model = make_ptncn(zeta=0.0)
    # U1 should not exist
    assert not hasattr(model, "U1")

def test_in_dim_defaults_to_x_dim():
    model = PTNCN(x_dim=VOCAB, hid_dim=HID)   # no in_dim
    # check model.in_dim == VOCAB
    assert model.in_dim == VOCAB
    # check model.M1.shape[0] == VOCAB
    assert model.M1.shape[0] == VOCAB

def test_forward_output_shape():
    model = make_ptncn()
    B = 4
    x = torch.zeros(B, IN_DIM)
    mask = torch.ones(B, 1)
    z0_o_logits, z0_o = model(x, mask)
    assert z0_o_logits.shape == (B, IN_DIM)
    assert z0_o.shape == (B, IN_DIM)

def test_forward_state_initialized_after_first_step():
    model = make_ptncn()
    B = 4
    x = torch.zeros(B, IN_DIM)
    mask = torch.ones(B, 1)
    model(x, mask)
    assert model.y1 is not None
    assert model.y2 is not None
    assert model.e1y is not None
    assert model.e2y is not None
    
def test_mask_zeroes_errors():
    # mask = 0 for all samples → e1 and e0 should be all zeros
    model = make_ptncn()
    B = 4
    x = torch.randn(B, IN_DIM)
    mask = torch.zeros(B, 1)       # all padding
    model(x, mask)
    assert torch.all(model.e1 == 0)
    assert torch.all(model.e0 == 0)

def test_lra_shift_on_second_step():
    # after t=0, zf1_tm1 at t=1 should equal y1 from t=0 (not zf1)
    model = make_ptncn()
    B = 2
    x = torch.randn(B, IN_DIM)
    mask = torch.ones(B, 1)
    model(x, mask)
    y1_from_t0 = model.y1.clone()
    model(x, mask)                 # second step
    assert torch.allclose(model.zf1_tm1, y1_from_t0)


def test_compute_updates_changes_weights():
    # weights should change after calling compute_updates
    model = make_ptncn()
    B = 2
    x = torch.randn(B, IN_DIM)
    mask = torch.ones(B, 1)
    model(x, mask)          # t=0: zf0_tm1 is zeros
    model(x, mask)          # t=1: zf0_tm1 = x from t=0 — now non-zero
    
    m1_before = model.M1.data.clone()   # clone after 2 passes
    w1_before = model.W1.data.clone()
    model.compute_updates()
    
    assert not torch.allclose(model.M1.data, m1_before)
    assert not torch.allclose(model.W1.data, w1_before)

def test_compute_updates_preserves_shapes():
    # shapes must not change after update
    model = make_ptncn()
    B = 4
    x = torch.ones(B, IN_DIM)
    mask = torch.ones(B, 1)
    model(x, mask) # t=0: sets real state
    model(x, mask) # t=1: zf0_tm1 is now non-zero
    model.compute_updates()
    # now check M1 and W1 changed
    assert model.M1.shape == (IN_DIM, HID)
    assert model.W1.shape == (HID, IN_DIM)
    assert model.V1.shape == (HID, HID)
    assert model.E1.shape == (IN_DIM, HID)
    assert model.M2.shape == (HID, HID)
    assert model.W2.shape == (HID, HID)
    assert model.V2.shape == (HID, HID)
    assert model.E2.shape == (HID, HID)


def test_compute_updates_no_u1_when_zeta_zero():
    # should not crash and U1 should not exist
    model = make_ptncn(zeta=0.0)
    B = 4
    x = torch.ones(B, IN_DIM)
    mask = torch.ones(B, 1)
    model(x, mask)          # t=0: sets real state
    model(x, mask)          # t=1: non-zero tm1
    model.compute_updates() # this should not crash
    assert not hasattr(model, "U1")

def make_embed_ptncn(**kwargs):
    defaults = dict(vocab_size=VOCAB, emb_dim=EMB, hid_dim=HID)
    defaults.update(kwargs)
    return EmbeddingPTNCN(**defaults)

def test_embedding_output_shape():
    # token_ids: (B,) integers in [0, VOCAB)
    # output logits should be (B, VOCAB)
    model = make_embed_ptncn()
    B = 4
    token_ids = torch.randint(0, VOCAB, (B,))
    mask = torch.ones(B, 1)
    logits = model(token_ids, mask)
    assert logits.shape == (B, VOCAB)

def test_embedding_clear_state():
    # after forward then _clear_state:
    # model.ptncn.y1 should be None
    model = make_embed_ptncn()
    B = 4
    token_ids = torch.randint(0, VOCAB, (B,))
    mask = torch.ones(B, 1)
    model(token_ids, mask)
    model._clear_state()
    assert model.ptncn.y1 is None

def test_embedding_compute_updates_changes_weights():
    # 2 forward passes, then compute_updates
    # check ptncn.M1 changes
    model = make_embed_ptncn()
    B = 2
    token_ids = torch.randint(0, VOCAB, (B,))
    mask = torch.ones(B, 1)
    model(token_ids, mask)          # t=0: zf0_tm1 is zeros
    model(token_ids, mask)          # t=1: zf0_tm1 = x from t=0 — now non-zero
    
    m1_before = model.ptncn.M1.data.clone()   # clone after 2 passes
    w1_before = model.ptncn.W1.data.clone()
    model.compute_updates()
    
    assert not torch.allclose(model.ptncn.M1.data, m1_before)
    assert not torch.allclose(model.ptncn.W1.data, w1_before)
