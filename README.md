# Mortgage Default Prediction with Machine Learning | Group 17

Research seminar project based on Barbaglia, Manzan and Tosetti (2021), *Forecasting Loan Default in Europe with Machine Learning*, *Journal of Financial Econometrics*. This is an adaptation using US Freddie Mac data, not a replication of the authors' original results: their European DataWarehouse data were not available to us.

## Group members

- [Chistiakova Maria](https://github.com/mariachistiakova) — part 1, data preparation
- [Rudko Sofia](https://github.com/Sophie-Rudko) — part 2, replication of the paper
- [Meyzler Liya](https://github.com/kodallline) — part 3, future default and regional calibration
- [Samoilova Maria](https://github.com/merysamoylova) — part 4, independent evaluation

## Project

We use the Freddie Mac Single-Family Loan-Level Dataset: 450,000 fixed-rate mortgages originated in 2010–2018 and observed in 2013–2018 in 50 states and DC. As in the paper, we compare penalized logistic regression, gradient boosting and XGBoost with one model per state and a random 60/20/20 split of loans (protocol R). As our extension (protocol E), we train on 2013–2014, forecast first 90+ day defaults over the next three months in 2017–2018, and compare a global calibration with a regional (state-level) calibration with shrinkage, including a fixed budget of reviewing the top 1% of loans each quarter.

Boosting ranks loans better than logistic regression in most states (mean within-state ROC AUC gain +0.042 for gradient boosting and +0.040 for XGBoost), which matches the direction of the paper. Part of this gain comes from our calculated current LTV; without it the gain is +0.019 and +0.026. In the future period ranking quality falls (AUC 0.791 → 0.702) and all versions predict only about half of the observed defaults. Regional calibration slightly lowers log loss (the bootstrap interval is below zero), but at a 1% review budget it finds the same 78 defaults as the global calibration. Our LTV is a proxy, there is no income or employment data, and one hurricane quarter holds 37% of the test defaults. See the notebook for methods, figures, results and further limitations.

## Files

| File / folder | Purpose |
| --- | --- |
| `notebooks/Group17_Mortgage_Final.ipynb` | Main notebook (parts 1–4), including saved outputs |
| `src/data_preparation/` | Part 1 data pipeline as Python scripts, with tests |
| `results/part1_data_preparation/` | Sample sizes, splits, data checks and figures of the prepared data |
| `results/part2_replication/` | Validation and test scores for 51 states, robustness table, tuning plans |
| `results/part3_forecasting/` | Frozen procedure, base model, calibration choices and test results |
| `results/part4_evaluation/` | Final metrics, bootstrap intervals, top-1% review budget and figures |
| `presentation/Group17_Mortgage_Presentation.pdf` | Slides |
| `requirements.txt` | Library versions from our Colab runs |
| `.gitignore` | Excludes data and locally generated files |

## Input files

The input files are **not** in this repository: they contain Freddie Mac loan-level data and are larger than GitHub allows. Ask us for access to our Google Drive folder with them.

| File | Size | Needed? | What it is |
| --- | --- | --- | --- |
| `participant_1_data_preparation.zip` | 330 MB | **Required** | Prepared data from part 1: protocols R and E, regional features, config and checks |
| `Participant_4_Evaluation_R.zip` | — | Recommended | Our saved part 2 results (the `mortgage_part2` folder). Part 2 then skips the state models that are already trained, and part 4 reads the test predictions from it |

Do not rename them: the notebook looks for these exact names and checks the SHA-256 of the data. Part 3 creates `participant_3_outputs.zip` itself, and part 4 uses it in the same session.

## Run in Google Colab

1. Download `notebooks/Group17_Mortgage_Final.ipynb` and open it in [Google Colab](https://colab.research.google.com/) (File → Upload notebook). A standard CPU runtime is enough.
2. Upload `participant_1_data_preparation.zip` and `Participant_4_Evaluation_R.zip` to the Colab session's `/content` directory (Files sidebar → Upload). Wait until both uploads have finished.
3. Select **Runtime → Run all**. The notebook installs missing packages, unpacks the data, checks it and runs parts 1–4 in order.
4. Part 2 is the slow part. Without `Participant_4_Evaluation_R.zip`, all 51 state models are trained from scratch, which takes a few hours, and part 4 then packs the new results itself.
5. In Colab the browser downloads `participant_3_outputs.zip` and `participant_4_results.zip` at the end of parts 3 and 4; this is expected. Save a copy of the notebook after a successful run. Colab uploads are temporary, so upload the input files again in a new session.

The notebook also runs locally (e.g. VS Code): set `PROJECT_PATH` in the settings cell to the unpacked `data_preparation` folder and install `requirements.txt`. `MODE = "rebuild"` rebuilds the prepared data from the raw Freddie Mac sample files (`RAW_SOURCE_ROOT`), the same as `src/data_preparation/run_pipeline.py`.

## Reference

Barbaglia, L., Manzan, S., & Tosetti, E. (2021). Forecasting Loan Default in Europe with Machine Learning. *Journal of Financial Econometrics*. https://doi.org/10.1093/jjfinec/nbab010

Data: Freddie Mac Single-Family Loan-Level Dataset, https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

The paper PDF is not redistributed in this repository.
