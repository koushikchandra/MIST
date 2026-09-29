# Baselines

The four baselines MIST is compared against in the paper, all **feature-matched**: each consumes
the *same* frozen patch features, splits, per-fold gene panel, `log1p` normalization, and
`pearson_mean` metric as MIST. Only the spatial-modeling core differs — each method's original
raw-image front-end is replaced by the shared patch feature, and its published modeling core is
kept.

| `--model` | Method | Core | Reference |
|-----------|--------|------|-----------|
| `stnet`   | ST-Net  | per-spot MLP regression, no spatial context | He et al. 2020 |
| `hist2st` | Hist2ST | transformer (global) + GraphSAGE over kNN(coords) (local) + JK-LSTM | Zeng et al. 2022 |
| `bleep`   | BLEEP   | bi-modal contrastive image/expression + kNN retrieval at inference | Xie et al. 2023 |
| `stem`    | STEM    | conditional DDPM with a DiT denoiser | Zhu et al. 2025 |

All baseline code is in `baseline_spatial.py`. It is self-contained: `stflow/` and `evaluation.py`
here vendor the only helpers it needs, so no external project is required.

## Reproduce

Run from **this** directory (so `stflow/` and `evaluation.py` are importable). Data is resolved by
the same convention as MIST (see the top-level README §2):

```bash
# one baseline, one regime, one seed
python baseline_spatial.py --model stnet --regime LOOO --seed 1 \
    --splits_root  /path/to/cross_organ_splits \
    --source_dataroot /path/to/dataset \
    --embed_dataroot  /path/to/embed_dataroot \
    --feature_encoder uni_conch \
    --save_root results_baselines --device 0
```

`--regime` is one of `LOOO`, `POOLED`, `INTRA`. To reproduce every paper baseline, sweep the four
models across the regimes and seeds 1, 2, 3, then average across seeds (the paper reports
`mean ± std` over seeds). Model-specific knobs (all have paper defaults): `--k_retrieval` /
`--bleep_batch` / `--max_ref` for BLEEP, `--n_steps` for STEM, `--depth2` / `--depth3` / `--k` for
Hist2ST.

Each run writes `<save_root>/<REGIME>_<model>_seed<seed>/` containing `fold_<f>_results.json`,
`results_kfold.json` (aggregate, with `pearson_mean`), and per-fold `test_predictions.npz`.

## Quick demo

The bundled demo (`../demo_data/`) includes a cross-organ split so you can smoke-test any baseline
end-to-end in seconds, CPU is fine:

```bash
python baseline_spatial.py --model bleep --regime LOOO --seed 1 \
    --splits_root ../demo_data/cross_organ \
    --source_dataroot ../demo_data/source \
    --embed_dataroot  ../demo_data/embed \
    --feature_encoder uni_v1_official \
    --save_root results_demo --device 0 --epochs 3
```

This is a smoke test on 400-spot crops, not a benchmark — the PCC is not meaningful.
