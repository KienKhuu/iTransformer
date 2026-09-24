# Contrastive Learning integration

This package is meant to be copied into the existing project.

## Replace these files
- `main.py`
- `model/iTransformer.py`
- `model/PatchTST.py`
- `model/TSMixer.py`
- `model/LSTM.py`

## Add this folder
- `contrastive/`
  - `augmentations.py`: weak jitter + jitter/masking views
  - `losses.py`: projection head + symmetric NT-Xent/InfoNCE
  - `trainer.py`: contrastive pretraining loop
  - `__init__.py`

No `*_cl.py` model files are created. The original `forward()` and `forecast()` methods are left unchanged. Each forecasting model only gains an `encode()` method used during contrastive pretraining.

## Pipeline executed by main.py
For every model and seed:

1. Create one random model initialization.
2. Clone it into `baseline_model` and `cl_model` so they start from identical weights.
3. Train/evaluate `baseline_model` with the existing supervised forecasting pipeline.
4. Contrastively pretrain `cl_model` using two augmented views of the same stock window and InfoNCE.
5. Discard the temporary projection head.
6. Fine-tune `cl_model` using the exact same existing supervised `train_model()` function.
7. Evaluate both on the same test split and print MAE/RMSE plus relative MAE change.

## Default CL hyperparameters
- CL epochs: 20
- projection dimension: 64
- temperature: 0.2
- jitter std: 0.02 (inputs are StandardScaler-normalized)
- masking ratio: 0.10
- CL learning rate: 1e-3

These are starting values, not claimed optimal values. Tune them on the validation set only; do not use the test set for hyperparameter selection.

## Representation used by each backbone
- iTransformer: mean pool original variate tokens after its encoder, before the forecasting projector.
- PatchTST: mean pool variables and patches after the PatchTST encoder, before the flatten forecasting head.
- TSMixer: flatten the representation after all mixer residual blocks, before the temporal forecast projection.
- LSTM: final recurrent output (`lstm_out[:, -1, :]`) before the forecasting FC layer.

## Test status
Synthetic smoke tests were run for all four models. Each retained its original forecast output shape and successfully completed contrastive pretraining through `encode()`.

The sandbox used for these checks does not have `yfinance` installed, so the full online AAPL data download was not rerun here. The supplied baseline already depends on `yfinance`; use the same environment in which your original baseline succeeds.
