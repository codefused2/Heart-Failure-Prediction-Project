# Heart Failure Mortality Prediction

A machine learning project exploring whether baseline clinical measurements can be used to predict mortality among patients with heart failure.

This was my first attempt at taking a classification problem through the full machine learning workflow rather than stopping at `model.fit()` and an accuracy score.

The goal was not just to build the highest-scoring model possible, but to understand how model evaluation, data leakage, cross-validation, hyperparameter tuning, threshold selection, and final testing should actually be handled.

---

## The Problem

The dataset contains clinical records for 299 patients with heart failure.

The target variable is:

- `DEATH_EVENT = 0` → patient survived during the follow-up period
- `DEATH_EVENT = 1` → patient died during the follow-up period

The objective of this project was to predict mortality using information that would be available at the patient's initial assessment.

---

## Dataset

The dataset contains clinical variables such as:

- Age
- Ejection fraction
- Serum creatinine
- Serum sodium
- Creatinine phosphokinase (CPK)
- Platelet count
- Anaemia
- Diabetes
- High blood pressure
- Smoking status
- Sex

The original dataset also contains a variable called `time`.

`time` represents the patient's follow-up duration rather than a baseline clinical measurement. Since the goal of this project is to make a prediction using information available at the initial assessment, this feature was excluded to avoid using post-baseline information.

After removing the target and `time`, 11 features were used for modeling.

---

## Exploratory Data Analysis

The exploratory analysis included:

- Dataset structure and descriptive statistics
- Missing-value inspection
- Target class distribution
- Feature distributions
- Correlation analysis
- Outlier inspection

Several variables contained highly skewed distributions and extreme values, particularly:

- Creatinine phosphokinase
- Serum creatinine
- Platelet count

These observations were not automatically removed because extreme clinical measurements may represent genuine patient conditions rather than data errors.

A `log1p` transformation was applied to CPK and serum creatinine to reduce strong right-skewness while keeping all observations.

---

## Modeling Approach

The data was first divided into training and held-out test sets using a stratified split.

The test set was kept separate during model development and was used only for the final evaluation.

Several classification algorithms were initially compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

Models that require feature scaling were implemented using scikit-learn pipelines so that preprocessing was performed correctly inside the cross-validation process.

Because the dataset is small, model performance was evaluated using stratified cross-validation rather than relying on a single validation split.

The strongest baseline candidates were:

- Logistic Regression
- Random Forest
- Support Vector Machine

These models were then optimized using Optuna.

---

## Why F1-Score?

The target classes are moderately imbalanced, and identifying patients with `DEATH_EVENT = 1` is particularly important.

Accuracy alone could favor the majority class, while optimizing recall alone could produce too many false-positive predictions.

For that reason, F1-score was used as the main optimization metric because it balances precision and recall.

---

## Tuned Model Comparison

After hyperparameter optimization, the three final candidates produced very similar cross-validation results.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 76.09% | 61.71% | 72.03% | **65.93%** | **78.80%** |
| Random Forest | **76.91%** | **64.67%** | 67.07% | 65.24% | 78.07% |
| SVM | 75.42% | 60.37% | **73.32%** | 65.70% | 78.44% |

Since F1-score was the primary model-selection metric, Logistic Regression was selected as the final model.

Its slightly higher ROC-AUC, simpler structure, and greater interpretability also supported the decision.

The differences between the models were small, so these results should not be interpreted as evidence that Logistic Regression strongly outperformed the alternatives.

---

## Threshold Optimization

A classifier does not directly produce a final `0` or `1`. Logistic Regression first produces a probability for the positive class.

The usual classification threshold is `0.50`, but there is no guarantee that this is the most suitable threshold for every problem.

To avoid optimizing the threshold on the test set, out-of-fold predictions were generated from the training data using stratified cross-validation.

Different thresholds were then evaluated based on precision, recall, and F1-score.

The threshold that produced the highest training out-of-fold F1-score was:

**0.48**

At this threshold:

| Metric | Score |
|---|---:|
| Precision | 60.82% |
| Recall | 76.62% |
| F1-score | 67.82% |

The threshold was then fixed before evaluating the final test set.

---

## Final Evaluation

After all model-development decisions were completed, Logistic Regression was trained on the full training set and evaluated on the held-out test set.

| Metric | Test Score |
|---|---:|
| Accuracy | 66.67% |
| Precision | 47.62% |
| Recall | 52.63% |
| F1-score | 50.00% |
| ROC-AUC | 75.35% |

The confusion matrix showed:

- 30 survivors correctly classified
- 10 deaths correctly classified
- 11 survivors incorrectly classified as deaths
- 9 deaths incorrectly classified as survivors

The final performance was noticeably lower than the cross-validation performance.

Rather than tuning the model again using the test results, the test performance was kept as-is. The test set is intended to estimate generalization to unseen data, not to become another dataset for model optimization.

---

## What I Learned

One of the most useful parts of this project was seeing that a good cross-validation result does not guarantee an equally strong result on a held-out test set.

With only 299 patients, performance estimates can change considerably depending on which observations appear in a particular split.

The project also helped me understand why the machine learning workflow matters just as much as the model itself.

In particular:

- A feature can contain useful predictive information and still be inappropriate because it would not exist at prediction time.
- The test set should not be repeatedly used to make modeling decisions.
- Cross-validation gives a more reliable view of model behavior than a single validation split.
- Accuracy alone can be misleading for an imbalanced classification problem.
- Classification thresholds affect precision and recall and can be treated as part of the model-development process.
- A weaker final result is more useful than an artificially inflated result produced by repeatedly tuning against the test set.

---

## Limitations

This project has several important limitations.

The dataset contains only **299 patients**, which makes performance estimates unstable and limits how confidently the results can be generalized.

Mortality was also treated as a binary classification problem even though patients were observed for different lengths of time. A proper survival-analysis approach would be better suited for directly modeling time-to-event outcomes.

The held-out test set contains only around 60 patients, meaning that only a few prediction changes can noticeably affect metrics such as recall and F1-score.

Finally, this model was built as an educational machine learning project. It has not been clinically validated and should not be used for medical decision-making.

---

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Optuna
- Jupyter / Google Colab

---

## Project Workflow

```text
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Preprocessing
        ↓
Train / Test Split
        ↓
Baseline Model Comparison
        ↓
Cross-Validation
        ↓
Hyperparameter Optimization
        ↓
Tuned Model Comparison
        ↓
Final Model Selection
        ↓
Threshold Optimization
        ↓
Held-Out Test Evaluation
