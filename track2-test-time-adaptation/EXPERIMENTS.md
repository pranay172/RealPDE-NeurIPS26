# Experiments — Track 2

Every experiment in the order I ran them, dead ends included. Local results come from leakage-safe splits that hold out whole Reynolds-number and angle-of-attack regimes. The main split is 56 training trajectories plus 25 held out, and a complementary fold holds out a different set. Scores marked **official** are the development-leaderboard results I recorded from Codabench.

Only the code and configs behind the best submission are in this repo.

| What I tried | Outcome |
|---|---|
| Environment bootstrap | CPU and GPU environments, starter-kit smoke tests passing |
| Pinned official container | Offline stream, reset, bounds and checkpoint checks passing |
| Data audit | Found the duplicate trajectory; built leakage-safe split manifests |
| Persistence and all released baselines | Picked the packed FNO as the base |
| Split-safe FNO fine-tune | First leakage-safe learned baseline |
| One-step full-model SGD at test time | Spent runtime, no meaningful gain |
| Interval calibration on training data only | Better bounds, same point predictions |
| First submission package | **Official 75.42** |
| Last-layer SGD at test time | Same story: runtime cost, no meaningful gain |
| Longer FNO training | Too little gain for triple the budget |
| Complementary holdout fold | Recipe confirmed on a different regime split |
| Channel-balanced / wake-weighted loss | Hurt Rel-L2 and MVPE |
| Coverage-targeted static interval | Better local SPS, same points |
| Score-aligned loss | TKE up, but Rel-L2 down |
| Train-scaled official loss | Missed the Rel-L2 and MVPE gates |
| Refit on all 81 valid trajectories | Done |
| All-data refit packaged | **Official 75.50** |
| Online residual-scale uncertainty | Worse than the static interval |
| Blending FNO output with the current input | Failed the TKE gate |
| Error diagnosis | Most error was in the fluctuations, which pointed to the dual-head design |
| Dual-head FNO + streaming mean-bias correction | Accepted. After the loader-bug fix below, its exact starting weights couldn't be confirmed |
| Full-data dual-head model | Trained from **random** weights because of a checkpoint-loading bug (found later) |
| Full-data dual-head model packaged | **Official 76.50** |
| Per-pixel variance EMA ("emaV") | Better local TKE |
| emaV package | Platform failure: Codabench couldn't fetch the ingestion bundle |
| Linear variance head on a frozen trunk | Worse calibration than emaV |
| Coverage feedback, input blending | Failed gates |
| Joint simulation + real fine-tuning | TKE collapsed |
| Hardened emaV package | **Official 76.76** (TKE +6.6 over the previous package) |
| Pooled AR(2) correction | +0.08 Rel-L2, −0.21 TKE. Found and fixed the loader bug |
| Proper simulation initialization | Better on every quality metric in all four fold/seed pairs (TKE +2.08 on average) |
| Stride-20 sampling, 1800 updates | Confirmed on both folds and seeds |
| Both of the above on all data, same wrapper | **Official 77.62 — best** |
