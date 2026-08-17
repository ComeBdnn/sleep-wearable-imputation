# Sleep Wearable Imputation

Missing data imputation for Oura ring sleep data, subjective sleep rating and cognition test scores (N-Back test), tested on a dataset of 29 participants, cumulating over a thousand days of data recorded.

Method: mask known values -> impute -> evaluate against the hidden truth, using k-fold cross-validation and three metrics (nRMSE, nBias, variance retention).

## Findings

| | Best method | Notes |
|---|---|---|
| HR/ HRV / Breathing Rate | Kalman Smoothing | Beats the mean, the comparison baseline on accuracy and also preserves more variance |
| Sleep Stages Durations | TabPFN | Only method that clearly beats the mean baseline |
| Subjective Sleep Rating | All methods fall near the mean baseline | Looks like a ceiling effect for our approach |
| N-Back Score | All methods fall near the mean baseline | Consequence of the almost zero correlation between sleep/cognition |

### Additional Notes

- Every accuracy gain on the sleep stages durations columns comes with more variance loss (sd_ratio < 1)
- SAITS (deep learning model, not pre-trained): unremarkable nRMSE, but large nBias compared to the other methods, probably an exemple of the real cost of training on too little data.
- TimesFM implementation tried out, but running time on personal GPU is too long, interesting to be tested on a GPU server, but will probably face the same problems as other models of the same sort.

## Repository Structure

```
├── notebooks/
│   └── cleared_imputation.ipynb   # main pipeline: masking, methods, evaluation
├── results_cache/                 # cached per-method results (no recomputing unless changes are made)
├── figures/                       # exported heatmaps
├── data/
│   ├── README.md
│   ├── oura_sleep.csv
│   └── selfreport_nback_data.csv
├── downstream/
│   ├── data/                      # exported filled datasets with each method
│   ├── README.md
│   └── main_randomforest_compare.py   # feature prediction from imputed data
├── docs/
│   └── project_summary.md
├── requirements.txt
└── requirements-additional.txt
```

## Setup

```
pip install -r requirements.txt
```

Additional (Needed to run SAITS and TabPFN from scratch):
```
pip install -r requirements-additional.txt
```
TabPFN needs extra setup...

## Reproducing Results

Open `notebooks/imputation.ipynb` and run all. The results for each metrics are cached per method in `results_cache/` (to limit the runtime, a method is recomputed if it is modified, or, of course, a new method is added), so running a fresh clone of the repository produces the same findings above without recomputing all methods.


## Downstream

`downstream/` predicts N-Back score and self assessment rating from the dataset, to test wether imputation method choice changes downstream predictions, results from the base datasets with gaps and different filled datasets are to be compared.

## Data

See `data/README.md` for file descriptions and provenance.

## Limitations