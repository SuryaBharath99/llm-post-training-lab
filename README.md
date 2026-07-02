# llm-post-training-lab

> A hands-on lab for **LLM post-training**: SFT, preference optimization (DPO & friends),
> and RL with verifiable rewards (GRPO / RLVR). Each method is a small, runnable experiment
> on a single GPU, instrumented so the **failure modes** (reward hacking, KL blow-up, format
> collapse) are visible — because *explaining what broke* is what proves hands-on RL skill.

## Why this repo

Reading about GRPO is not the same as running it. The goal here is to **close the loop**:
take a small model, define a **verifiable reward**, run a real training loop, measure
before/after, and document what actually happened. The first milestone is the canonical
entry point to modern reasoning RL — **GRPO on GSM8K with an exact-match verifier (RLVR)**.

## The method ladder (what I'm working through)

Each rung fixes the previous one's pain and exposes a new gap:

| # | Method | Signal | What it buys |
|---|--------|--------|--------------|
| 0 | **SFT** | demonstrations | instruction-following / format |
| 1 | **DPO** (+ KTO/ORPO/SimPO) | preference pairs | alignment without a reward model or RL loop |
| 2 | **RLVR + GRPO** | deterministic verifier | reasoning gains, no learned reward model, no critic |
| 3 | **PRM vs ORM** | step vs outcome reward | reward the *reasoning*, not just the answer |
| 4 | **Agentic / multi-turn RL** | trajectory outcome | long-horizon tool use (later) |

The constant across all of them: a **KL leash to the reference model** that keeps the policy
from drifting into the reward-hacking corner.

## Repo structure

```
llm-post-training-lab/
├── README.md                       # this file — plan + experiment index
├── requirements.txt
├── common/                         # shared, reused across experiments
│   ├── verifier_gsm8k.py           # parse final answer + exact-match reward
│   ├── eval.py                     # before/after accuracy on a held-out split
│   └── logging_utils.py            # reward / KL / length logging helpers
├── experiments/
│   ├── 01_grpo_gsm8k_rlvr/         # ← first milestone
│   │   ├── train_grpo.py           # TRL GRPOTrainer loop
│   │   ├── config.yaml             # model, group size, KL beta, LoRA, etc.
│   │   └── FINDINGS.md             # what broke + how I'd fix it at scale
│   ├── 02_dpo_contrast/            # DPO on a preference set (later)
│   └── 03_prm_vs_orm/             # step vs outcome reward (later)
└── azure/                          # GPU provisioning + env setup notes
```

## Experiment index

| ID | Method | Task | Model | Status |
|----|--------|------|-------|--------|
| 01 | GRPO (RLVR) | GSM8K (exact-match) | Qwen2.5-1.5B-Instruct | 🚧 planned |
| 02 | DPO | preference pairs | Qwen2.5-1.5B-Instruct | ⬜ backlog |
| 03 | PRM vs ORM | GSM8K reasoning | Qwen2.5-1.5B-Instruct | ⬜ backlog |

## Milestone 01 — GRPO on GSM8K (RLVR)

**Goal:** fine-tune `Qwen2.5-1.5B-Instruct` with GRPO where reward = *is the final answer
exactly correct?* — no reward model, no human labels.

**Build steps**
1. **Verifier reward** — parse `#### <answer>`, exact-match → `1.0`, else `0.0`; small
   format bonus for producing the expected structure.
2. **Baseline eval** — GSM8K test accuracy *before* training (the honest "before").
3. **GRPO loop** — TRL `GRPOTrainer`: sample `G` completions/prompt, group-normalize the
   advantage `Â = (r − mean)/std`, clipped PPO-style update + KL to reference.
4. **After eval** — same held-out split; compare accuracy.
5. **Instrument** — log **reward**, **KL-to-reference**, **mean completion length**, accuracy.
6. **Break it & document** — length hacking, format gaming, KL runaway (lower β), reward
   plateau, format collapse → written up in `FINDINGS.md`.

**Why Qwen2.5-1.5B-Instruct:** strong math prior (non-flat reward signal to learn from),
Apache-2.0, first-class TRL support, and already formats answers so the focus stays on the
RL loop.

## Environment (Azure GPU)

Single-GPU runs. Config is chosen by available VRAM:

- **≥ 40 GB** → full fine-tune, group size `G = 8`.
- **16–24 GB** → **QLoRA** (4-bit base + LoRA adapters), `G = 4–6`, vLLM for fast rollouts.

> GPU SKU / VRAM: _TBD — fill in from `nvidia-smi` on the Azure box, then finalize `config.yaml`._

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login        # model + dataset access
wandb login                  # optional: metric logging
```

## Findings-first philosophy

Every experiment ends with a `FINDINGS.md`: what the curves did, what broke, and how it
would be mitigated at scale. That artifact — not the training script — is the point.

## References

- Shao et al., *DeepSeekMath (GRPO)*, 2024 — https://arxiv.org/abs/2402.03300
- DeepSeek-AI, *DeepSeek-R1*, 2025 — https://arxiv.org/abs/2501.12948
- Rafailov et al., *DPO*, 2023 — https://arxiv.org/abs/2305.18290
- Lightman et al., *Let's Verify Step by Step (PRM)*, 2023 — https://arxiv.org/abs/2305.20050
- HuggingFace TRL — https://github.com/huggingface/trl