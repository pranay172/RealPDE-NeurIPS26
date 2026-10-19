# Usage — Track 2

Run every command from inside `track2-test-time-adaptation/`, with the environment from the [top-level setup](../README.md#-setup) activated.

## 1. Restore the data and starter kit

None of these are included in the repository.

**Data.** Download the competition release from [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data) into the repo root. You need `train_real/` and `baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth`. Don't use `train_real/7575_0.h5`, which duplicates another trajectory. The split manifests already leave it out.

**Starter kit.** Sign in to [Codabench competition 17385](https://www.codabench.org/competitions/17385/), download **starting kit v6** from the Files tab, and extract it here as `realpde_t2_starting_kit_v6/`. It must contain:

```
realpde_t2_starting_kit_v6/
├── load_baseline.py
├── scoring.py
├── local_eval.py
├── pack_ckpt_fp16.py
├── rpde_baselines/
└── example_data/mean_std_real.pt
```

The final layout:

```
RealPDE-NeurIPS26/
├── RealPDE-Competition-Data/
│   ├── train_real/*.h5
│   └── baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth
└── track2-test-time-adaptation/
    ├── realpde_t2_starting_kit_v6/
    └── ...
```

## 2. Train

```bash
python scripts/train_dual_head_fno.py --config configs/train.json --skip-eval
```

This converts the simulation FNO into the dual-head model, checks the conversion tensor by tensor, then trains for 1800 updates on all 81 valid trajectories. It took about 6 minutes on an RTX 5050. The packed checkpoint goes to `artifacts/dual_head_fno/fno_dual_head_fp16.pth`, and a `training_report.json` sits next to it. Add `--smoke` for a two-update test run.

## 3. Package and check

`configs/package.json` guards against packaging the wrong weights: it holds the SHA-256 of the checkpoint to package, and the packager refuses any other bytes. Hash your trained checkpoint:

```bash
sha256sum artifacts/dual_head_fno/fno_dual_head_fp16.pth
```

Paste the hash into `checkpoint_sha256` in `configs/package.json`, in place of the placeholder. Leave the adaptation and interval settings alone. Then build the zip and check it:

```bash
python scripts/package_submission.py --config configs/package.json --skip-full-eval --device cpu
python scripts/check_deployed_parity.py \
  --submission artifacts/package/extracted \
  --output artifacts/package/parity_report.json
```

The zip lands in `artifacts/package/submission.zip`. The packager runs the kit's `local_eval.py` on the extracted zip. It then runs it again inside the organizers' Docker image, unless you pass `--skip-docker`. The parity check replays six real windows from two trajectories through the packaged wrapper and the research predictor, and fails if they differ by more than 2e-6.

## Not re-run

I haven't rerun this pipeline on the real data since the competition ended, and retraining won't reproduce byte-identical weights. Before publishing, I smoke-tested every step above on stand-in data. Treat the steps above as the recipe, not a guaranteed path back to 77.62.
