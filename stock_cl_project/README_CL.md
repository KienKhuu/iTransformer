# Contrastive pretraining extension

This project keeps the four forecasting architectures as the baseline source of truth and adds a shared contrastive-pretraining pipeline.

## Models
- iTransformer: mean pooling of original variate tokens after Transformer encoder.
- PatchTST: mean pooling of `[n_vars, patch_num]` axes after the patch Transformer encoder.
- TSMixer: flattened representation after all mixer residual blocks and before forecasting projection.
- LSTM: last hidden output before the final fully connected forecasting layer.

## CL training
1. Create two views of the same standardized stock window:
   - view 1 = weak Gaussian jitter
   - view 2 = weak Gaussian jitter + random value masking
2. Send both views through the **same forecasting backbone** (`encode`).
3. Send each representation through a temporary 2-layer projection head.
4. Optimize symmetric NT-Xent / InfoNCE with in-batch negatives.
5. Discard the projection head.
6. Fine-tune the same forecasting model with the original supervised Close-price MSE pipeline.

## Run
Install the same dependencies used by the baseline, then from this directory:

```bash
python main.py
python main_cl.py
```

`main_cl.py` can also run paired baselines from identical initial weights (`RUN_PAIRED_BASELINES=True`) and saves results to `results/contrastive_results.json`.
