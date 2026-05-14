# data-analysis-tool

A robust Python toolkit designed for data cleaning, exploration, and interactive visualization within Google Colab environments. This tool automates common preprocessing tasks such as data sanitization, missing value imputation, outlier detection, and advanced statistical association mapping.

## Features

-   **Intelligent Data Loading**: Automatically handles common null strings (e.g., '?', 'N/A', 'NULL') and attempts auto-conversion of columns to correct numeric types.
-   **Comprehensive Inspection**: Quickly view data dimensions, column type breakdowns, and statistical summaries for both numerical and categorical data.
-   **Automated Cleaning**: 
    -   Identify and impute missing values using mean, median, mode, or constant strategies.
    -   Remove exact duplicate rows and specific outliers using IQR logic.
    -   Interactive row and column deletion.
-   **Advanced Scaling & Encoding**:
    -   **Numeric**: Min-Max, Standard (Z-score), and Robust scaling.
    -   **Categorical**: One-Hot, Ordinal, and Uniform encoding.
-   **Interactive Visualizations**: Powered by Plotly, including horizontal violin plots, scatter plots, histograms, and grouped bar charts.
-   **Deep Statistical Insights**: 
    -   Pearson Correlation heatmaps for numeric data.
    -   Cramér's V heatmaps for categorical associations.
    -   Unified Association Heatmaps combining Numeric and Categorical data (using Point-Biserial and Eta/ANOVA).

## Installation

```bash
# Basic installation
pip install "git+https://github.com/mugalan/data-analysis-tool.git"

# Install with plotting support (required for visualizations)
pip install data-analysis-tool[plotting]
```

## Quick Start (Use Cases)

The tool is optimized for use in **Google Colab**.

### 1. Data Cleaning and Imputation
Load a dataset and handle missing values in one flow.

```python
from data_analysis import DataInspector

inspector = DataInspector()

# Step 1: Upload your CSV (Interactively in Colab)
inspector.upload_data()

# Step 2: Impute missing values using the median strategy
inspector.handle_missing_values(strategy='median')

# Step 3: Remove duplicate rows
inspector.remove_duplicates()
```

### 2. Exploratory Data Analysis (Visual)
Generate multi-chart statistical views of your variables.

```python
# Visualize numerical distributions (Violin, Scatter, and Histogram in one view)
inspector.plot_numerical(['Age', 'Salary'])

# Explore relationships between two variables
# (Auto-selects Scatter, Box, or Grouped Bar based on types)
inspector.plot_relationship('Department', 'Salary')
```

### 3. Feature Engineering & Normalization
Prepare your data for Machine Learning models.

```python
# Scale numeric columns using Robust Scaling (better for outliers)
normalized_numeric = inspector.extract_normalized_numeric_data(method='robust')

# Encode categorical columns using One-Hot encoding
encoded_cat = inspector.extract_normalized_categorical_data(method='onehot')

# Create a single merged DataFrame ready for training
final_df = inspector.create_normalized_data_df()
```

### 4. Advanced Correlation Mapping
Identify hidden associations across different data types.

```python
# Generate a unified heatmap for both numeric and categorical columns
inspector.plot_all_associations_heatmap()
```

## Project Structure

```text
data-analysis-tool/
├── data_analysis/
│   ├── __init__.py
│   └── core.py        # Contains DataInspector and PlottingMethods
├── pyproject.toml     # Project configuration
└── README.md
```

## Authors
-   **Mugalan** - [mugalan@gmail.com](mailto:mugalan@gmail.com)

## License
This project is licensed under the MIT License.
