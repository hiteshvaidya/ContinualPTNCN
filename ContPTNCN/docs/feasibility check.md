# Integrating fast weights into P-TNCN: a complete NeurIPS 2026 research plan

**The proposed integration of fast weights into P-TNCN fills a genuine gap in the literature — no existing work combines fast weight mechanisms with predictive coding networks.** This creates a compelling novelty angle: bio-plausible, locally-trained fast weight programming where prediction errors serve as the "programming instructions" for rapid synaptic modulation. The research is well-timed given the 2025–2026 surge in papers connecting fast weight programmers to neurobiology (Irie & Gershman, TMLR 2026; Gershman, Fiete & Irie, *Neuron* 2025). However, the student's current baseline and benchmark plan has critical gaps that must be addressed before submission.

---

## (a) The student's baseline set has three critical gaps

The proposed comparison against standard PTNCN, GRU, LSTM, and xLSTM is a reasonable start but will draw immediate reviewer objections. NeurIPS reviewers in the bio-plausible/local learning space consistently expect a **two-tier comparison structure**: Tier 1 compares against other bio-plausible/local learning methods (the "fair" comparison), and Tier 2 includes BPTT-trained models as performance ceiling references. The student's plan is missing the entire first tier.

**Essential baselines (7 models — will face rejection without these):**

| Model | Learning rule | Bio-plausible? | Why essential |
|-------|-------------|----------------|---------------|
| Standard PTNCN (LRA) | Local (rec-LRA) | Yes | Ablation — shows what fast weights add |
| Vanilla/Elman RNN | BPTT | No | Simplest recurrent baseline |
| LSTM | BPTT | No | Universal sequence baseline |
| GRU | BPTT | No | Standard BPTT companion |
| **e-prop** (Bellec et al., 2020, *Nature Comms*) | Local eligibility traces | Yes | **Most critical missing baseline** — the established bio-plausible RNN learning rule |
| **xLSTM** (Beck et al., NeurIPS 2024 Spotlight) | BPTT | No | mLSTM's covariance update is *directly related to fast weight programmers* |
| **Mamba-1 or Mamba-2** (Gu & Dao, 2023/2024) | End-to-end gradient | No | Dominant efficient sequence model; reviewers will ask if omitted |

**Recommended additions (include 2–3 of these):**

The **Linear Recurrent Unit** (Orvieto et al., ICML 2023) bridges SSMs and RNNs and is increasingly used in bio-plausible RNN papers — the RTU paper at NeurIPS 2024 builds directly on it. A **Forward-Forward or Predictive Forward-Forward** variant provides another bio-plausible reference point (and Ororbia's own PFF work at CogSci 2023 makes this natural to include). **RWKV-7** ("Goose," March 2025) uses a generalized delta rule with vector-valued gating that connects to fast weight programmers, making it relevant but primarily tested at LLM scale. RetNet is subsumed by the Mamba comparison and can be safely excluded.

**The xLSTM inclusion is strategically excellent.** The mLSTM variant uses a covariance update rule that the xLSTM paper explicitly acknowledges as related to Schmidhuber's Fast Weight Programmers (1992). This creates a clean **2×2 comparison matrix**: {fast weights vs. no fast weights} × {local learning vs. BPTT}. PTNCN+FastWeights occupies the unique quadrant of bio-plausible local learning *with* fast weight memory — no other model in the comparison set does this.

**Fairness framing in the paper:** Recent NeurIPS/ICML papers on local learning (Counter-Current Learning, NeurIPS 2024; ModProp, NeurIPS 2022; RTU, NeurIPS 2024; SOFO, NeurIPS 2024) all include BPTT-trained models as "performance ceiling" references while clearly acknowledging the comparison crosses learning paradigms. The established framing is: "We include BPTT-trained models as performance upper bounds. Our goal is to demonstrate that PTNCN+FastWeights narrows the gap between local learning and end-to-end gradient optimization." This is expected and accepted by reviewers.

---

## (b) Penn Treebank alone won't cut it — here's the benchmark suite

The student's proposed benchmarks (PTB, enwik8, text8, Sequential MNIST, Permuted MNIST, Split CIFAR) are a mix of still-useful and outdated choices. **PTB is now widely considered a legacy benchmark** — its 929K training words with artificially constrained 10K vocabulary make it unsuitable as a primary evaluation. WikiText-103 (~103M words, realistic vocabulary) has replaced it as the standard word-level language modeling benchmark and is used by S4, Mamba, RWKV, and virtually all modern sequence modeling papers.

**Recommended benchmark suite:**

**Tier 1 — Essential (must include all five):**

**WikiText-103** is the standard for comparing SSMs, RNNs, and Transformers at word level. **Multi-Query Associative Recall (MQAR)** from Zoology (Arora et al., NeurIPS 2024) is *the* diagnostic for fast weight mechanisms — it tests in-context key-value retrieval where standard SSMs notably underperform Transformers. If PTNCN+FastWeights performs well on MQAR, this becomes a headline result. **Sequential MNIST / Permuted MNIST** (784-length sequences) remains essential for bio-plausible papers because e-prop, RFLO, ModProp, and virtually all competing methods report on it, enabling direct comparison. **Copy task and adding problem** at varying lengths directly test what fast weights should excel at: sequence memory and selective attention over long horizons. **Split CIFAR-100** (20 tasks × 5 classes) is the workhorse continual learning benchmark and demonstrates catastrophic forgetting resistance.

**Tier 2 — Strongly recommended (include 2–3):**

**enwik8** (character-level, 100MB) remains appropriate for recurrent models and provides a byte-level evaluation free from tokenization assumptions. **Long Range Arena** is still expected for efficient sequence model papers, though a January 2025 analysis demonstrated that most LRA performance comes from short-range dependencies — include it but acknowledge the limitations, and emphasize Path-X (16K tokens) as the genuine long-range test. **TIMIT phoneme classification** is essential if comparing against e-prop, which reports strong results on this task. **Permuted MNIST extended to 200 sequential tasks** specifically tests the kind of continual learning where local/Hebbian-like rules have a natural advantage — a 2025 paper on bio-inspired metaplasticity uses exactly this protocol.

**Where local learning has a natural advantage:** The compelling benchmarks for this paper are those testing **online/continual learning** (where BPTT requires storing full history but local rules operate in O(1) memory), **associative recall** (where fast weights provide quadratic storage capacity vs. linear for hidden states), and **non-stationary data streams** (where catastrophic forgetting is the primary failure mode). MQAR and the 200-task Permuted MNIST protocol are the two benchmarks most likely to produce results reviewers find compelling.

**Avoid relying solely on:** PTB (outdated), The Pile at scale (requires compute beyond available GPUs for meaningful comparison to Mamba/Pythia scaling curves), or MNIST-only evaluations (too easy for modern methods).

---

## (c) Fast weights literature reveals a clear gap for PTNCN

The fast weights lineage spans nearly four decades, from Hinton & Plaut's dual-weight concept (1987) through Schmidhuber's Fast Weight Programmers (1992), Ba et al.'s modern revival (NeurIPS 2016), and a remarkable recent convergence with attention mechanisms and neurobiology. **No existing work combines fast weights with predictive coding networks or P-TNCN**, making this a genuine contribution.

**The foundational arc:** Hinton and Plaut (1987) introduced the idea that each connection has two weights — a slowly changing "plastic" weight storing long-term knowledge, and a fast-changing "elastic" weight storing temporary knowledge that decays toward zero. Schmidhuber (1992) formalized this into the Fast Weight Programmer (FWP): a "slow" network learns to produce context-dependent weight changes for a "fast" network using outer products. Ba, Hinton, Mnih, Leibo & Ionescu (NeurIPS 2016) revived the concept for modern RNNs with a Hebbian outer-product update rule: **A(t) = λA(t-1) + η·h(t)·h(t)ᵀ**, where an inner loop iteratively refines the hidden state using the fast weight matrix. This achieved state-of-the-art results on associative retrieval tasks with small networks.

**The transformer connection:** Schlag, Irie & Schmidhuber (ICML 2021, 264 citations) proved that **linearized self-attention is equivalent to Fast Weight Programmers**. Keys and values in attention correspond to outer product programming instructions. They proposed a delta rule update to correct key-value mappings, improving over purely additive Hebbian updates. This paper unified the fast weights and attention literatures. Irie, Schlag, Csordás & Schmidhuber (NeurIPS 2021) extended this to Recurrent Fast Weight Programmers, adding recurrence to both slow and fast networks. The Self-Referential Weight Matrix (ICML 2022) pushed further — a neural network whose weight matrix modifies itself.

**The 2024–2026 explosion:** DeltaNet (Yang et al., NeurIPS 2024) made fast weight programmers scalable via hardware-efficient parallel training, with 1.3B parameter models outperforming Mamba. Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) combined gating for adaptive memory erasure with the delta rule for precise memory modifications, outperforming Mamba-2 and entering production systems (Qwen3-Next, Kimi Linear). **Most importantly for this project, Irie & Gershman (TMLR 2026) published the definitive bridge paper** connecting FWPs to neurobiology, reviewing FWPs as 2D-state RNNs with matrix-form hidden states interpretable as time-varying synaptic weights. Gershman, Fiete & Irie (*Neuron*, 2025) established key-value memory as a computational principle in the brain, with Hebbian learning implementing writes and pattern completion implementing reads.

**The gap PTNCN+FastWeights fills:** Five specific gaps exist in the literature. First, no bio-plausible fast weights in predictive coding — P-TNCN would be the first. Second, Ba et al. (2016) used BPTT for slow weights but Hebbian rules for fast weights; P-TNCN could use purely local rules for *both*. Third, predictive coding errors could serve as "programming instructions" for fast weights, creating a more principled and biologically motivated update rule than arbitrary outer products. Fourth, P-TNCN already excels at continual learning, and fast weights could further enhance this by storing recent context without interfering with long-term knowledge. Fifth, the fast/slow weight distinction directly implements complementary learning systems theory (McClelland, McNaughton & O'Reilly, 1995) using bio-plausible local rules.

**Must-cite papers:** Hinton & Plaut (1987), Schmidhuber (1992), Ba et al. (NeurIPS 2016), Schlag, Irie & Schmidhuber (ICML 2021), Yang et al. DeltaNet (NeurIPS 2024), Yang et al. Gated DeltaNet (ICLR 2025), Irie & Gershman (TMLR 2026), Gershman, Fiete & Irie (*Neuron* 2025), and the neuroscience foundations: McClelland et al. (1995) on complementary learning systems, Zucker & Regehr (2002) on short-term synaptic plasticity, Benna & Fusi (*Nature Neuroscience* 2016) on multi-timescale synaptic dynamics. A 2025 eLife reviewed preprint on "Fast and Slow Synaptic Plasticity" describes almost exactly what PTNCN+FastWeights should do: fast synaptic changes greedily suppress errors using real-time feedback while slow changes implement statistically optimal learning.

---

## (d) Both GPUs are more than sufficient — VRAM is a non-issue

The P-TNCN at the planned scale (3 layers, hidden dim 512, character-level vocab ~256) is roughly **~2.6M parameters** — three orders of magnitude smaller than models that stress modern GPUs. Total VRAM requirements including parameters, gradients, optimizer states, activations, and framework overhead amount to approximately **1.5–2 GB in FP32**, leaving **~14 GB free on the RTX 4080** and ~46 GB free on the L40.

| Specification | RTX 4080 | L40 | L40/4080 ratio |
|---|---|---|---|
| VRAM | 16 GB GDDR6X | 48 GB GDDR6 (ECC) | 3.0× |
| FP32 TFLOPS | 48.7 | 90.5 | 1.86× |
| Memory bandwidth | 717 GB/s | 864 GB/s | 1.21× |
| Tensor Cores | 304 (4th gen) | 568 (4th gen) | 1.87× |

**Estimated training times:**

| Dataset | RTX 4080 | L40 |
|---------|----------|-----|
| PTB (char-level) | **2–6 hours** | 1–4 hours |
| enwik8 (100MB) | **12–48 hours** | 8–30 hours |
| Sequential MNIST | **1–4 hours** | 0.5–2.5 hours |
| Online CL tasks | Minutes to hours | Minutes to hours |

The critical caveat is that **RNNs are inherently sequential across timesteps**, so GPU utilization for hidden=512 will be only **5–20% of peak TFLOPS** regardless of GPU choice. The workload is memory-bandwidth-bound, not compute-bound, which means the L40's practical speedup is closer to **~1.3–1.5×** rather than the theoretical 1.86×. The biggest performance lever is batch size — with 14 GB free, batch sizes of **512–2048** are easily feasible, which dramatically improves GPU utilization for small RNNs.

**Recommended GPU allocation:** Use the RTX 4080 for all development, debugging, PTB runs, Sequential MNIST, online CL tasks, and ablation studies. Reserve the L40 for enwik8/text8 full training runs and large hyperparameter sweeps. Run different configurations on both GPUs simultaneously to double throughput. Mixed precision training (`torch.cuda.amp`) provides ~1.5–2× speedup with no memory concerns. When porting to JAX, `jax.lax.scan` for the temporal loop and aggressive JIT compilation should yield an additional ~10–30% speedup. **Pre-load entire datasets to GPU memory** — PTB (5MB) and even enwik8 (100MB) fit trivially, eliminating all I/O bottleneck.

---

## (e) Adapting autoresearch for PTNCN requires three key changes

Karpathy's autoresearch is a ~630-line Python framework where an AI agent modifies `train.py`, trains for a fixed 5-minute window, evaluates against a single scalar metric, and keeps or discards the change via git. Over two days, Karpathy's agent ran ~700 experiments, found ~20 genuine improvements, and reduced time-to-GPT-2-quality by 11%. The pattern adapts cleanly to PTNCN with three structural changes.

**Change 1 — Replace the metric.** Swap `val_bpb` (bits-per-byte) with a PTNCN-appropriate metric: validation cross-entropy loss for language modeling, negative accuracy for classification tasks, or a composite temporal prediction error. The metric must be deterministic and comparable across architectural changes. Keep the "lower is better" convention.

**Change 2 — Seed domain knowledge into program.md.** The original autoresearch gives the agent generic ML optimization directions. For PTNCN, `program.md` must include specific research directions: temporal coding layer configurations, error correction mechanisms, parallel stream architectures, predictive coding variants, fast weight update rules (Hebbian outer product vs. delta rule vs. gated updates), learning rate schedules for dual-timescale systems, and normalization strategies. Without domain-specific seeding, the agent will perform generic hyperparameter search rather than meaningful architectural exploration.

**Change 3 — Adjust the time budget.** PTNCN training dynamics differ from LLM pretraining. Start with 5 minutes per experiment on PTB, but calibrate based on convergence behavior — the model may need longer to reveal meaningful differences between architectural choices. For enwik8, consider 15–20 minute windows.

**Running Claude Code as the orchestrator:** Install Claude Code CLI on the remote server via npm, authenticate using SSH port forwarding for the OAuth flow, then launch in a persistent tmux session with `claude --dangerously-skip-permissions`. The `CLAUDE.md` file should specify that only `train.py` is editable, list allowed commands (python, git, grep, nvidia-smi), and instruct the agent to never pause for permission. For overnight runs, add a watchdog script that monitors experiment count and GPU temperature, and cap at ~100 experiments to manage API costs (~$50–150 per night depending on model choice). All results accumulate in `results.tsv` and git history.

An alternative worth examining is **ARIS** (Auto-Research-In-Sleep), an open-source framework built specifically for autonomous ML research with Claude Code. It includes experiment-bridge skills with W&B logging, Hydra-based hyperparameter tuning, and cross-model collaboration — and has been used for accepted AAAI 2026 papers. The **Agentic Researcher** framework (arXiv 2603.15914) offers sandboxed container-based execution that scales to multi-GPU clusters. Either of these may provide more robust overnight operation than the raw autoresearch pattern.

---

## (f) NeurIPS 2026 considerations for bio-plausible local learning papers

**The local learning community at NeurIPS is growing rapidly.** Counter-Current Learning (NeurIPS 2024), ModProp (NeurIPS 2022), SOFO (NeurIPS 2024), RTU (NeurIPS 2024), and the least-control principle (NeurIPS 2022) form a clear lineage. Reviewers drawn from this pool will expect the paper to cite and position against these methods. The xLSTM (NeurIPS 2024 Spotlight) and Mamba-3 (ICLR 2026) are fresh enough that reviewers will be aware of them — omitting them risks appearing uninformed.

**The fast weight angle is uniquely strong.** The 2025–2026 papers by Irie, Gershman, and Fiete have created a direct ML-to-neurobiology bridge for fast weight programmers, published in *Neuron* and TMLR. This gives the project strong neuroscientific grounding beyond typical bio-plausible ML papers. Frame the contribution as implementing complementary learning systems theory with purely local rules: fast weights greedily suppress prediction errors in real-time while slow weights encode structural regularities through rec-LRA.

**Practical submission advice:** Structure the results tables with explicit columns for "Learning Rule," "Uses BPTT?," "Bio-Plausible?," and "Param Count" so reviewers can immediately see the comparison structure. Present results in two sections: (A) comparison with bio-plausible methods (the fair comparison) and (B) comparison with BPTT/gradient-trained methods (performance ceiling reference). The most compelling possible results are: (1) PTNCN+FastWeights approaching or matching BPTT-trained LSTM/GRU on language modeling while using only local learning, (2) outperforming standard SSMs on MQAR associative recall, and (3) demonstrating superior continual learning performance on the 200-task Permuted MNIST protocol. If the paper can show all three, it makes a strong NeurIPS case regardless of whether it matches Mamba on raw perplexity.

## Conclusion

The project occupies a genuinely novel intersection: fast weight programming (now understood as equivalent to linear attention) combined with predictive coding (a leading bio-plausible learning framework), implemented with purely local learning rules. The recommended baseline set of 7–10 models spanning both bio-plausible methods (e-prop, standard PTNCN) and BPTT references (LSTM, GRU, xLSTM, Mamba) matches NeurIPS community expectations. The benchmark suite should center on WikiText-103, MQAR, Sequential/Permuted MNIST, and Split CIFAR-100, with enwik8 and LRA as secondary evaluations. Both available GPUs are more than sufficient for training — the bottleneck is wall-clock time on enwik8 runs (~1–2 days), not memory. The autoresearch pattern provides a viable framework for automated exploration, particularly with ARIS or Claude Code as the orchestrating agent. The strongest strategic move is positioning MQAR associative recall as the "killer benchmark" — if PTNCN+FastWeights matches transformers on key-value retrieval where standard SSMs fail, using only local learning rules, that result alone justifies the paper.