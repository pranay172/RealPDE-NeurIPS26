# Experiments — Track 1

Every experiment from August–September 2026 in the order I ran them, dead ends included. **Fold A** and **Fold B** are my two local validation splits. Each holds out whole Reynolds-number and angle-of-attack regimes. Scores marked **official** are the development-leaderboard results I recorded from Codabench. Everything else is local.

Only the code and configs behind the best submission are in this repo.

| What I tried | Outcome |
|---|---|
| Environment bootstrap | CPU and GPU environments working |
| Verifier for the official evaluator container | Used as a pre-upload check from here on |
| Leakage-safe validation harness | Persistence baseline: Rel-L2 93.77, TKE 66.67, MVPE 93.82, Time 98.64, SPS 16.99 |
| Plain CNO fine-tune on real data | Collapsed on edge regimes |
| Post-mortem of that fine-tune | The Re 3750 regime was failing |
| Scale-balanced u/v loss | Best clean point predictor so far |
| Interval calibration on training data only (α = 0.025, β = 0.15) | SPS +22 |
| First package | Rejected: predictions depended on batch size |
| Batch-invariant, one-sample-at-a-time wrapper | **Official 74.84** |
| Out-of-fold calibration | Landed on the same bounds |
| Fold B robustness check | Recipe held up |
| All-data 600-update package | Official 74.19 (worse) |
| Three-fold comparator | Reference numbers for later gates |
| Persistence for the first 4 frames | **Official 75.57** |
| Zero-shot FNO | Rejected |
| Matching window scale | Rejected |
| Split field / mean / fluctuation / TKE loss | Rejected |
| Wake-weighted loss | Rejected |
| Frozen *simulation* CNO + adapter | Rejected |
| Real-adapted CNO + zero-initialized residual Conv3d adapter | New Fold A leader |
| Real-adapted CNO + adapter on Fold B | Robust |
| Real-adapted CNO + adapter packaged | **Official 75.87** |
| Heteroskedastic (learned-variance) head | Rejected |
| Three residual adapters averaged on one backbone | Helped on Fold A only |
| Three-adapter average packaged | ⏸️ Withheld |
| Three-adapter average on Fold B | Small gains, not submitted |
| MFDR + log-spectral loss | TKE up, but failed the other gates |
| Rollout curriculum for the adapter | New Fold A champion |
| Rollout curriculum on Fold B | Robust |
| All-data rollout package | **Official 75.88** |
| Spatial heteroskedastic uncertainty | Missed the SPS threshold |
| FiLM-conditioned adapter | Kept as an ensemble member |
| Multi-scale wavelet-TKE adapter | TKE 73.45 on Fold A; ensemble member |
| Three-adapter ensemble package | ⏸️ Passed checks, then quarantined after the ensemble collapse below |
| Longer backbone training, 600 → 3000 updates | Gains on every subscore (Fold A) |
| Longer backbone recipe on all data | Backbone for the next two packages |
| Three-adapter ensemble on the all-data long backbone | **Official 7.96.** Outliers amplified through the ensemble, plus a wrapper bug |
| All-data long backbone + one rollout adapter | **Official 77.31 — best** |
| Long backbone on Fold B | Confirmed: Rel-L2 95.56, TKE 75.39, MVPE 96.06, SPS 41.27 |
| Rollout adapters on the long backbones | Passed Fold A, failed the Fold B TKE gate |
| 6000 updates | ⏸️ Paused |
| Interval scaling by leave-one-regime-out | SPS +1.55 (A), +0.99 (B); exploratory |
| Locked audit protocol | ⏸️ Registered, never run |
| Fold B long backbone + conditional intervals | Submitted, failed evaluation (no log) |
