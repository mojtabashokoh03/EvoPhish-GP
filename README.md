# EvoPhish-GP: A Correlation-Aware Symbolic Meta-Ensemble Framework for Adaptive Phishing Website Detection

## Overview

This repository contains the source code and experimental implementations associated with the research paper:

**"EvoPhish-GP: A Correlation-Aware Symbolic Meta-Ensemble Framework for Adaptive Phishing Website Detection."**

The proposed framework generates **15 meta-features** for phishing website detection using a correlation-aware symbolic meta-ensemble approach.

This repository provides the datasets used in the experiments, the dataset loading implementation, and the scripts used for fitness evaluation, comparative experiments, and ablation studies.

## Datasets

Two datasets were evaluated in this study:

- **PWD2021:** The `PWD2021.csv` file contains the PWD2021 dataset.
- **ISCX-URL-2016:** The `AllISCX.csv` file contains the ISCX-URL-2016 dataset.

The datasets are loaded and accessed during the experiments using the following script:

```text
dataset_loader.py
```

Please ensure that the dataset files are placed in the appropriate locations expected by the dataset loader before running the experiments.

## Source Code and Experimental Scripts

The main scripts included in this repository are described below.

### 1. Fitness Evaluation

- **`cpfinal.py`** — Implements the single-objective fitness function used in the initial experimental implementation.
- **`cgpf1_final.py`** — Implements the subsequent single-objective fitness evaluation approach used in the experiments.

### 2. Comparative Experiments and Ablation Studies

- **`allcomp.py`** — Used for comparative experiments and related evaluations.
- **`allcompnew.py`** — Used for additional comparative experiments and ablation study analyses reported in the paper.

These scripts support the experimental evaluation and comparison of the proposed approach. Please refer to the corresponding source code and the research paper for implementation details and experimental configurations.

## Running the Experiments

Before running the scripts:

1. Ensure that Python is installed.
2. Install the required Python dependencies.
3. Place `PWD2021.csv` and `AllISCX.csv` in the locations expected by `dataset_loader.py`.
4. Run the relevant experimental script according to the evaluation you wish to reproduce.

For example, to execute a script, use:

```bash
python dcpfinal.py
```

**Note:** `dataset_loader.py` is responsible for loading the datasets. Run the appropriate experimental script separately to perform the corresponding evaluation. The exact execution commands and required arguments may depend on the implementation.

## Contact

For questions about the source code, experimental methodology, datasets, or research paper, please contact:

- **Mojtaba Shokoh:** [Mojtaba.shokoh03@gmail.com](mailto:Mojtaba.shokoh03@gmail.com)
- **Institutional email:** [shokoh@pgs.usb.ac.ir](mailto:shokoh@pgs.usb.ac.ir)

Researchers interested in reproducing the experiments or discussing the proposed EvoPhish-GP framework are welcome to contact us.

