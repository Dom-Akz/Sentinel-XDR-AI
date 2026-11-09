# Network Intrusion Detection - FastAPI + Streamlit

**Soufiane Moussaoui - Farnani Ferdaousse, encadrant: M. Moumen Idriss, 5CIRA-G1**

Upload a CSV of network flows, click **Analyze**, get an AI verdict per flow (Normal / Attack), the attack
category, a confidence score, a SHAP explanation, charts, and a downloadable result file.
Pure Python, runs locally, no cloud, no database. Academic, defensive project: it only reads dataset files,
it never touches a network.

```
User -> Streamlit dashboard -> FastAPI -> validation + preprocessing -> Random Forest -> prediction + SHAP -> dashboard
```

> **Honesty note.** This code was built and tested end to end (44 automated tests, live API, headless dashboard run)
> using **synthetic** data, because the real dataset is not downloadable from the build environment.
> **No result in this repository is a real UNSW-NB15 result.** Real scores appear only after _you_ train on the real files.
> How long training takes on the full dataset was not measured.

---

## 1. Dataset: UNSW-NB15

|                                                  |                                                                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Official page                                    | https://research.unsw.edu.au/projects/unsw-nb15-dataset                                                                                                |
| Official download (the "HERE" link on that page) | https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS                          |
| Files you need                                   | `UNSW_NB15_training-set.csv` (~175k flows) and `UNSW_NB15_testing-set.csv` (~82k flows), and optionally `UNSW-NB15_features.csv` (column descriptions) |
| Not needed                                       | the huge raw PCAPs and the four 2.5M-row CSVs                                                                                                          |
| License                                          | free for academic use; commercial use needs the authors' permission. **Cite the authors** (the papers are listed on the official page)                 |
| Account needed                                   | none (opens in a browser; the folder layout on the share may change, look for the exact file names)                                                    |

**Why this dataset.** It is public, labelled, tabular (flow statistics, no payload), comes with an _official_
train/test split, and has 9 attack families plus normal traffic, so it supports both "attack or not" and
"which attack" with a laptop CPU.

**Features (42).** Flow statistics computed from traffic: duration (`dur`), protocol / service / TCP state
(`proto`, `service`, `state`), packet and byte counts (`spkts`, `dpkts`, `sbytes`, `dbytes`), rates and loads,
TTLs, packet loss, jitter, inter-packet times, TCP window/RTT values, HTTP/FTP indicators, and "connection
counters" (`ct_*`: how many similar connections occurred recently). `id`, `attack_cat` and `label` are _not_
features. See `UNSW-NB15_features.csv` for the official descriptions.

**Labels (`attack_cat`).** `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`, `Analysis`,
`Backdoor`, `Shellcode`, `Worms`. `label` is the binary version (0 = normal, 1 = attack).

**Known weaknesses (say them in your report).** Lab-generated 2015 traffic with synthetic attacks; published
analyses report class overlap and labelling artefacts; some classes are tiny (Worms ~130 training rows).
Scores here do not predict real-world performance.

**Fallback mirrors** (unofficial; verify row counts and column names, cite the original authors):
Hugging Face `wwydmanski/UNSW-NB15` (parquet), Kaggle `dhoogla/unswnb15`, Hugging Face `Mireu-Lab/UNSW-NB15`.
`training/train.py` also accepts `.parquet` files, but they must contain the `attack_cat` column.

---
