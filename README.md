# Sleep Wearable Imputation

Missing data imputation for Oura ring sleep data, subjective sleep rating and cognition test scores, evaluated on 29 participants, cumulating 1877 nights.

Method: mask known values -> impute -> evaluate against the hidden truth, using repeated k-fold cross-validation and three metrics (normalized RMSE, normalized bias, variance retention).

## Findings

| | Best method | Notes |
|---|---|---|
| HR/ HRV / Breathing Rate | Kalman Smoothing | Beats the mean baseline on accuracy and preserves more variance |
| Sleep Stages Durations | TabPFN | Only method that clearly beats the mean baseline |
| Subjective Sleep Rating | All methods fall near the mean baseline | Looks like a ceiling effect for this approach |
| Cognition Test (N-back) Score | All methods fall near the mean baseline | Also looks like a ceiling effect for this approach |

### Additional Notes

- Every accuracy gain on the different sleep stages durations comes with more variance loss `sd_ratio << 1`
- SAITS (transformer trained from scratch, not a pretrained model) has unremarkable nRMSE results, but also a bias larger than any other method, likely the cost of training on only 29 participants.
- TimesFM is implemented, but runtime on CPU and personal GPU was too long to finish. Might be worth retrying on a dedicated GPU server, though it may hit the same limits the other models do.

## Repository Structure

```
├── notebooks/
│   └── imputation.ipynb               # main pipeline
├── results_cache/
├── figures/                           # exported heatmaps
├── data/
│   ├── README.md
│   ├── oura_sleep.csv
│   └── selfreport_nback_data.csv
├── requirements.txt
└── requirements-additional.txt
```

## Setup

```
pip install -r requirements.txt
```
This covers every method except TabPFN and TimesFM. Install those separately:
```
pip install -r requirements-additional.txt
```
**TabPFN** is pretrained and gated, installing the package is not enough on its own:

1. Create an account on [Prior Lab's website](https://ux.priorlabs.ai)
2. Go to the License tab and accept the license for the version of the model in use
3. Go to the account page and copy your API token
4. Set is as an environment variable:
```
export TABPFN_TOKEN=api_token_here
```
**TimesFM** only needs network access to Hugging Face to download its weights.


## Reproducing Results

Open `notebooks/imputation.ipynb` and run all. Each method's results are cached individually in `results_cache/`, so a method only recomputes if its own code changes or it's new, a fresh clone reproduces the findings above without recomputing anything.

## Data

See `data/README.md` for file descriptions and provenance.

## Limitations