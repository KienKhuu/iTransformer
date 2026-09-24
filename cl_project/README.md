# Contrastive pretraining for stock forecasting

This project adapts the four model files you uploaded. It adds a small causal
temporal encoder, overlapping masked views, cross-view temporal and instance
InfoNCE at four time scales, and frozen-encoder downstream comparisons. It is
**TS2Vec-inspired**, not a faithful TS2Vec reproduction or a new CL algorithm.

## Missing dependency from the uploads

The provided `iTransformer.py` and `PatchTST.py` import `layers.*`, but the
`layers/` package was not attached. Copy **your existing project's exact
`layers/` directory** beside `main.py` before running this package. Using a
different implementation would change the models being compared. The package
cannot run those two models from the attached files alone.

Install the project dependencies in your own Python environment:

```bash
python -m pip install torch numpy pandas scikit-learn yfinance
```

Check chronology, CL gradients, and causality before the full run:

```bash
python smoke_check.py
```

Run from this directory, with the matching `layers/` present:

```bash
python main.py --ticker AAPL --start 2015-01-01 --end 2026-01-01
```

Or run on a date-indexed OHLCV file that you used for your previous experiment:

```bash
python main.py --csv /path/to/aapl_ohlcv.csv --models PatchTST iTransformer
```

For a quick integration check, add `--cl-epochs 1 --epochs 2 --seeds 42`.
This quick check is not an experiment. The full run writes `results_cl.csv`.

## What the comparison measures

| Representation | Encoder | Forecast backbone | Final Close forecast |
| --- | --- | --- | --- |
| `raw` | none | one of the four models | last Close + predicted change |
| `random_frozen` | untrained, frozen TCN | same model | last Close + predicted change |
| `cl_frozen` | CL pretrained, frozen TCN | same model | last Close + predicted change |
| `persistence` | none | none | last Close |

The frozen encoder has output shape `[batch, seq_len, latent_dim]`. In the
encoded branches a trainable linear layer maps the backbone's `latent_dim`
outputs to one forecast of the Close change at each horizon. In the raw branch
the Close output is the forecast change. Both use the same last-price anchor.
The random and CL encoders start with identical weights for each seed.

Train/validation/test labels fall strictly within chronological 70/15/15 date
regions. Earlier context can be reused for a later target, as it would be
available in a live forecast. The scaler is fitted on the first 70% of dates
only; contrastive pretraining sees only training input windows. Model selection
uses validation MSE on standardized Close; MAE/RMSE are measured on held-out
test Close in original price units. The best validation checkpoint is restored
even if training reaches the epoch limit.

**Do not compare the new CSV directly with your earlier four numbers.** The
window boundaries, restored checkpoints, `auto_adjust=True` for fetched bars,
and common last-price residual forecast change the setup. Re-run `raw` in
this script for matched results. Check whether the models beat persistence.

## Code review of the supplied pipeline

1. `train_model` only restored the best state after patience was exhausted. If
   the last epoch finished before patience ran out, evaluation used the final
   weights instead of the validation-best weights.
2. The scaler was trained on the first 70% of **dates**, while the train/val
   boundary was computed on the first 70% of **windows**. These represent
   different dates, and some windows can have labels across the boundary.
3. The large LSTM error is not evidence the architecture cannot forecast:
   unlike the normalization in iTransformer and PatchTST, it had to learn
   changing absolute price levels from scratch. All new branches predict a
   change from the last observed Close. Diagnose the new matched result.
4. Your uploaded iTransformer and PatchTST are not standalone; the `layers/`
   package is required. The original script also needs `figure/` created
   before saving a plot. The new script reports CSV instead of plotting.
5. Two seeds give an unstable estimate of variability. After the first valid
   run, use at least five seeds if time permits, and report per-seed values.

## Limitations to state in the thesis

Nearby windows overlap substantially, so some temporal or batch negatives may
represent similar market states. CL has no guarantee of improving forecasts.
The `random_frozen` branch controls the encoder architecture, but encoded and
raw branches still use different output heads; treat `cl_frozen` versus
`random_frozen` as the cleanest attribution of a CL effect. One ticker and
one horizon do not establish generalization across markets or regimes. Do not
claim that better MAE implies a profitable trading strategy.
