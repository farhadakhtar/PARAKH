# Drop zone for Kaggle outputs

Claude has **no Kaggle access** (no CLI, no credentials, verified 2026-09-06),
so it cannot run the training or fetch the results. Drop them here instead and
they will be picked up automatically.

Download from `/kaggle/working/` after the run and place here:

    consistency_demo/metrics.json
    consistency_real/metrics.json        (if you ran on the real corpus)
    consistency_demo/consistency_model.keras
    consistency_demo/holdout_surprise.csv
    nlp_muril/metrics.json
    nlp_muril/work_text_heads.keras

Only `metrics.json` is needed to judge the round. The `.keras` files matter
for reuse, not for the verdict.
