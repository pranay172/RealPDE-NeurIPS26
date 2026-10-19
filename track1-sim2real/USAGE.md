# Usage — Track 1

Run every command from inside `track1-sim2real/`, with the environment from the [top-level setup](../README.md#-setup) activated.

## 1. Restore the data and starter kit

None of these are included in the repository.

**Data.** Download the competition release from [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data) into the repo root. You need `train_real/` and `baseline_checkpoints/sim_pretrain/`. Don't use `train_real/7575_0.h5`, which duplicates another trajectory. The split config already leaves it out.

**Starter kit.** Sign in to [Codabench competition 17363](https://www.codabench.org/competitions/17363/), download **starting kit v9** from the Files tab, and extract it here as `realpde_t1_starting_kit_v9/`. The code imports the CNO from it, so it must contain:

```
realpde_t1_starting_kit_v9/
├── load_baseline.py
├── scoring.py
├── rpde_baselines/
└── _vendor/einops/
```

The final layout:

```
RealPDE-NeurIPS26/
├── RealPDE-Competition-Data/
│   ├── train_real/*.h5
│   └── baseline_checkpoints/sim_pretrain/sim_cno.pth
└── track1-sim2real/
    ├── realpde_t1_starting_kit_v9/
    └── ...
```

## 2. Train the backbone

```bash
python scripts/train_backbone.py --device cuda
```

This fine-tunes the simulation CNO for 3000 updates on all 81 valid trajectories (Adam 3e-4 with cosine decay, batch 4, FP32). It took about 3 hours on an RTX 5050. The output goes to `artifacts/backbone/` as `final.pth` plus `normalizer.json`. Add `--smoke` for a two-update test run.

## 3. Train the adapter

```bash
python scripts/train_adapters.py --device cuda
```

This freezes the backbone from step 2 and trains three adapter variants, 600 updates each: rollout curriculum, FiLM and wavelet-TKE. Together they took about 10 minutes. Everything is saved to `artifacts/adapters/ensemble_payload.pth`, with keys `backbone_state_dict`, `member{1,2,3}_adapter_state`, `member{1,2,3}_adapter_config` and `normalizer`.

> [!CAUTION]
> Don't submit the three-member ensemble. It scored 7.96 on the server. My best submission uses **member 1 (rollout curriculum) only**.

## 4. Assemble the submission checkpoint

The wrapper in `submission/submission.py` loads a single `model.pth` next to itself: the backbone plus member 1, with `backbone.*` and `adapter.*` keys. No further training is involved. I did this merge during the competition but never committed it as a script. This snippet produces the same keys and shapes as my original checkpoint, and the wrapper loads its output with `strict=True`:

```python
import torch

payload = torch.load("artifacts/adapters/ensemble_payload.pth", map_location="cpu", weights_only=False)
state = {f"backbone.{k}": v for k, v in payload["backbone_state_dict"].items()}
state.update({f"adapter.{k}": v for k, v in payload["member1_adapter_state"].items()})
torch.save(
    {"model_kind": "residual_cno", "adapter_config": payload["member1_adapter_config"], "model_state_dict": state},
    "submission/model.pth",
)
```

The normalization constants hard-coded at the top of the wrapper are statistics of the 81 training trajectories, so the same data gives the same values. Check that they match your `artifacts/backbone/normalizer.json`.

Zip `submission.py`, `metadata` and `model.pth` together with the kit's `load_baseline.py`, `rpde_baselines/` and `_vendor/`, all at the root of the archive.

## Not re-run

I haven't rerun this pipeline on the real data since the competition ended, and retraining won't reproduce byte-identical weights. Before publishing, I smoke-tested every step above on stand-in data. Treat the steps above as the recipe, not a guaranteed path back to 77.31.
