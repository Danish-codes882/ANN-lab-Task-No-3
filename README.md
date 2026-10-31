# Single-Layer Perceptron from Scratch (NumPy)

**Name:** Muhammad Danish Wahab
**Student ID:** 24108244
**Program:** BS Artificial Intelligence (Batch 4C)
**Department:** Robotics & Artificial Intelligence
**Institute:** SZABIST Islamabad
**Email:** 24108244@szabist-isb.pk

---

## Project Overview

This assignment implements a **Single-Layer Perceptron** from scratch using only Python and NumPy (no prebuilt machine learning models) to solve a binary classification problem. The workflow covers data loading, exploration, preprocessing, training, testing, and evaluation.

## Dataset

| Item | Detail |
|---|---|
| Name | Blood Transfusion Service Center |
| Source | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center |
| DOI | https://doi.org/10.24432/C5GS39 |
| Creator | I-Cheng Yeh (2008) |
| Rows | 748 |
| Features | Recency, Frequency, Monetary, Time |
| Target | `Donated_Blood` (1 = donated in March 2007, 0 = did not) |
| Class split | 570 non-donors (0), 178 donors (1) |
| Missing values | None |

Note: the CSV in this folder was obtained from the OpenML mirror of the same UCI dataset. The target labels were converted from 1/2 to 0/1 and the columns were renamed to match the UCI feature names.

## Files

| File | Description |
|---|---|
| `blood_transfusion.csv` | Dataset |
| `perceptron_blood_transfusion.ipynb` | Complete code with outputs |
| `README.md` | This file |

## Preprocessing Steps

1. Checked shape, data types, and summary statistics
2. Checked missing values (none found) and duplicate rows (215 found, kept because they are different donors)
3. Checked correlations and dropped `Monetary` (it is exactly 250 x `Frequency`, correlation = 1.0)
4. Shuffled the data and split it 80% train / 20% test
5. Standardized features using the training set mean and standard deviation

## Model

- Single-layer perceptron with step activation function
- Weights and bias initialized to zero
- Learning rate: 0.01
- Epochs: 100
- Update rule: `w = w + lr * (target - prediction) * x`

## Results (Test Set, 150 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.82 |
| Precision | 0.78 |
| Recall | 0.22 |
| F1 Score | 0.34 |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 116 | 2 |
| **Actual 1** | 25 | 7 |

Recall is low because the data is imbalanced (only about 24% donors) and a single-layer perceptron can only learn a linear decision boundary.

## How to Run

1. Install the requirements:
   ```
   pip install numpy pandas matplotlib jupyter
   ```
2. Keep `blood_transfusion.csv` and the notebook in the same folder.
3. Open `perceptron_blood_transfusion.ipynb` in Jupyter Notebook or VS Code.
4. Run the cells one by one from top to bottom.
