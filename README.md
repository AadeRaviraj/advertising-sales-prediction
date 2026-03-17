# Advertising Sales Prediction — Linear Regression Case Study

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-Linear%20Regression-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete end-to-end machine learning project that predicts product sales based on advertising budgets spent across three channels — TV, Radio, and Newspaper. The project uses **Multiple Linear Regression** and follows a structured 13-step pipeline from data loading to visual evaluation.

---

## Problem Statement

Given the amount of money spent on advertising across TV, Radio, and Newspaper channels, can a machine learning model accurately predict the resulting sales figure?

This is a **regression problem** — the output is a continuous numeric value (sales amount), not a category.

---

## Project Workflow

| Step | Description |
|------|-------------|
| 1 | Load the dataset from a CSV file |
| 2 | Remove unwanted columns (e.g. `Unnamed: 0`) |
| 3 | Check for missing values |
| 4 | Display statistical summary of the dataset |
| 5 | Calculate correlation between all columns |
| 6 | Define Independent (X) and Dependent (Y) variables |
| 7 | Split data into Training (80%) and Testing (20%) sets |
| 8 | Create and train the Linear Regression model |
| 9 | Run predictions on test data |
| 10 | Evaluate model — MSE, RMSE, and R² Score |
| 11 | Display model coefficients and intercept |
| 12 | Compare Actual vs Predicted sales values in a table |
| 13 | Plot Actual vs Predicted sales scatter graph |

---

## Dataset

**File:** `Advertising.csv`

**Features (Independent Variables - X):**
- `TV` — advertising budget spent on TV (in thousands)
- `radio` — advertising budget spent on Radio (in thousands)
- `newspaper` — advertising budget spent on Newspaper (in thousands)

**Target (Dependent Variable - Y):**
- `sales` — product sales generated (in thousands of units)

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Multiple Linear Regression |
| Train/Test Split | 80% / 20% |
| Random State | 42 |
| Library | scikit-learn |

---

## Why Correlation Analysis?

Before training, the correlation matrix is calculated to understand the relationship between each advertising channel and sales. A higher correlation value means that channel has a stronger influence on predicting sales. This helps justify which features are worth including in the model.

---

## Evaluation Metrics

Since this is a regression problem, classification metrics like accuracy do not apply. Instead the model is evaluated using:

| Metric | Description |
|--------|-------------|
| MSE (Mean Squared Error) | Average of squared differences between actual and predicted values |
| RMSE (Root Mean Squared Error) | Square root of MSE — same unit as the target variable, easier to interpret |
| R² Score | How well the model explains the variance in sales. Closer to 1.0 is better |

---

## Model Coefficients

After training, the model outputs a coefficient for each feature. This tells us how much sales increase for every unit increase in that advertising channel's budget, while keeping other channels constant.

Example output format:
```
TV        : 0.054
radio     : 0.107
newspaper : 0.003
Intercept : 2.938
```

---

## Visualization

**Scatter Plot — Actual vs Predicted Sales**

A scatter plot is generated comparing actual sales values against the model's predicted values. Ideally, all points should fall close to a straight diagonal line, indicating accurate predictions.

---

## Tech Stack

- Python 3
- pandas — data loading, cleaning, and analysis
- numpy — RMSE calculation
- matplotlib — scatter plot visualization
- scikit-learn — model building, training, and evaluation

---

## How to Run

1. Clone this repository
2. Place `Advertising.csv` in the same folder as the script
3. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
4. Run the script:
   ```bash
   python AdvertisingCaseStudyModelBuildingVisualization..py
   ```

---

## Key Concepts Covered

- Supervised Machine Learning
- Regression (predicting continuous values)
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Multiple Linear Regression
- Model Coefficients and Intercept Interpretation
- Regression Evaluation Metrics (MSE, RMSE, R² Score)
- Actual vs Predicted Visualization

---

## Suggested Repo Name

`advertising-sales-prediction`

---

## Author

**Raviraj Aade**

Built as part of a Machine Learning Case Study series to understand regression problems, correlation analysis, and model evaluation using real-world advertising data.
