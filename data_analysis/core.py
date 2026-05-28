from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError, field_validator

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway,  multivariate_normal
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import FactorAnalysis




class DataInspector:
    """
    A comprehensive data cleaning and exploration tool for Google Colab.
    Provides interactive visualizations using Plotly and robust data sanitization.
    """

    def __init__(self):
        self.df = None
        self.numeric_df = None
        self.categorical_df = None
        self.categorical_normalized_df = None
        self.normalized_data_df = None
        self.numeric_normalized_df = None

    def upload_data(self):
        """
        Prompts user to upload a CSV, handles common null strings,
        and attempts to auto-convert columns to their correct numeric types.
        """
        uploaded = files.upload()
        if not uploaded:
            return print("No file uploaded.")

        file_name = list(uploaded.keys())[0]
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]),
                            na_values=['?', 'n/a', 'N/A', 'NULL', 'null', ' '])
        self.df['count']=1

        for col in self.df.columns:
            # Attempt to convert the column to numeric, forcing errors to NaN
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')

            # If the conversion didn't turn the entire column into NaNs
            # (and it wasn't already all NaN), we apply the change.
            if not numeric_col.isna().all():
                self.df[col] = numeric_col

        print(f"\n✅ File '{file_name}' loaded and types sanitized!")

    def get_summary(self):
        """
        Prints data dimensions and column type breakdown.
        Displays the first 20 rows of the DataFrame.
        """
        if self.df is None: return print("Error: No data loaded.")

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print(f"--- Data Summary ---")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")
        print(f"Numerical ({len(num_cols)}): {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")
        display(self.df.head(20))

    def show_missing_data(self):
        """
        Filters the DataFrame to show only rows containing at least one missing (NaN) value.
        """
        if self.df is None: return
        missing_mask = self.df.isnull().any(axis=1) | (self.df == "").any(axis=1)
        missing_rows = self.df[missing_mask]

        if missing_rows.empty:
            print("✨ No missing data found!")
        else:
            print(f"🔍 Found {len(missing_rows)} rows with missing values:")
            display(missing_rows)

    def delete_rows(self):
        """
        Deletes rows based on a comma-separated list of indices provided via user input.
        """
        if self.df is None: return
        try:
            user_input = input("Enter row indices to delete (e.g., 1, 3, 15): ")
            indices_to_drop = [int(i.strip()) for i in user_input.split(',') if i.strip().isdigit()]

            existing_indices = [i for i in indices_to_drop if i in self.df.index]
            self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
            print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def delete_columns(self):
            """
            Deletes columns based on a comma-separated list of names provided via user input.
            """
            if self.df is None:
                return print("No data loaded.")

            try:
                print(f"Current columns: {', '.join(self.df.columns)}")
                user_input = input("Enter column names to delete (e.g., Column1, Column2): ")

                # Split and clean the input
                cols_to_drop = [c.strip() for c in user_input.split(',')]

                # Filter to keep only columns that actually exist in the DataFrame
                existing_cols = [c for c in cols_to_drop if c in self.df.columns]

                if not existing_cols:
                    return print("⚠️ None of the provided column names were found.")

                # Drop the columns
                self.df = self.df.drop(columns=existing_cols)
                print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining count: {len(self.df.columns)}")

            except Exception as e:
                print(f"❌ Error: {e}")

    def handle_missing_values(self, columns=None, strategy='median', fill_value=None):
        """
        Imputes missing values in specified columns to preserve data rows.

        Parameters:
        - columns: List of strings. If None, applies to all columns with NaNs.
        - strategy: 'mean', 'median', 'mode', or 'constant'.
        - fill_value: Used only if strategy is 'constant'.
        """
        if self.df is None: return
        target_cols = columns if columns else self.df.columns[self.df.isnull().any()].tolist()

        for col in target_cols:
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)

        print(f"🛠️ Imputation complete using '{strategy}' strategy for: {target_cols}")

    def remove_duplicates(self):
        """
        Identifies and removes exact duplicate rows from the DataFrame to prevent statistical bias.
        """
        if self.df is None: return
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(self.df)
        print(f"✨ Removed {dropped} duplicate rows. New row count: {len(self.df)}")

    def export_cleaned_data(self, filename='cleaned_data.csv'):
        """
        Converts the current state of the DataFrame to a CSV file and
        triggers a browser download in the Google Colab environment.
        """
        if self.df is None: return
        self.df.to_csv(filename, index=False)
        files.download(filename)
        print(f"💾 '{filename}' has been generated and download triggered.")

    def column_details(self):
        """
        Iterates through all columns to show numeric ranges or categorical unique value counts.
        """
        if self.df is None: return
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"🔹 {col} (Numeric): Range [{self.df[col].min()} to {self.df[col].max()}]")
            else:
                print(f"🔸 {col} (Categorical): {self.df[col].nunique()} unique values")

    def get_categorical_summary(self):
        """
        Generates a detailed statistical summary for categorical columns,
        including unique counts, the most frequent value (Mode), and its frequency.
        """
        if self.df is None: return
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("No categorical columns found.")

        summary = cat_df.describe().T[['unique', 'top', 'freq']]
        print("--- Categorical Deep Dive ---")
        display(summary)

    def extract_numeric_data(self):
        """
        Filters the DataFrame to include only numeric columns and triggers a download.
        """
        if self.df is None: return print("Error: No data loaded.")

        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df

    def extract_categorical_data(self):
        """
        Filters the DataFrame to include only categorical (non-numeric) columns and triggers a download.
        """
        if self.df is None: return print("Error: No data loaded.")

        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df

    def extract_normalized_numeric_data(self, method='minmax'):
        """
        Extracts numerical columns and scales them using the specified method.

        Parameters:
        - method: str, options are:
          * 'minmax': Scales features exactly to the [0, 1] range. 
                      Best for algorithms that assume a bounded range (e.g., Neural Networks).
          * 'standard': Centers features to a mean of 0 and standard deviation of 1.
                        Standard choice for PCA, Clustering, and Linear models.
          * 'robust': Uses the median and Interquartile Range (IQR). 
                      Best if your data has outliers that you don't want distorting the scaling.
        """
        if self.df is None: return print("Error: No data loaded.")

        # Select only numerical columns
        num_df = self.df.select_dtypes(include=[np.number]).copy()

        if num_df.empty:
            print("⚠️ No numerical columns found to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df

        # Handle any missing values before scaling (fills with median if not handled yet)
        if num_df.isnull().any().any():
            print("ℹ️ Missing values detected. Imputing with column medians before scaling...")
            num_df = num_df.fillna(num_df.median())

        method_lower = method.lower().strip()

        # --- Option 1: Min-Max Scaling [0, 1] ---
        if method_lower == 'minmax':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)

        # --- Option 2: Standard Scaling (Z-score normalization) ---
        elif method_lower == 'standard':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)

        # --- Option 3: Robust Scaling (Handles Outliers gracefully) ---
        elif method_lower == 'robust':
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)

        else:
            print(f"❌ Unknown scaling method '{method}'. Defaulting to 'minmax'.")
            return self.extract_normalized_numeric_data(method='minmax')

        print(f"✨ Successfully scaled numerical data using the '{method_lower}' method.")
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method='uniform'):
        """
        Extracts categorical columns and applies the specified encoding method.
        
        Parameters:
        - method: str, options are:
          * 'uniform': Maps categories to numeric codes scaled to the [0, 1] range.
          * 'ordinal': Converts categories to distinct integers (0, 1, 2...) using OrdinalEncoder.
          * 'onehot': Converts categories to multiple binary (0 or 1) columns using OneHotEncoder.
          * 'minmax_ordinal': First encodes ordinally, then scales to exactly [0, 1] using MinMaxScaler.
        """
        if self.df is None: return print("Error: No data loaded.")

        # Select only categorical columns
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()

        if cat_df.empty:
            print("⚠️ No categorical columns found to transform.")
            self.categorical_normalized_df = pd.DataFrame()
            return self.categorical_normalized_df

        method_lower = method.lower().strip()

        # --- Option 1: Original Uniform Mapping ---
        if method_lower == 'uniform':
            for col in cat_df.columns:
                codes = cat_df[col].astype('category').cat.codes
                max_code = codes.max()
                if max_code > 0:
                    cat_df[col] = codes / max_code
                else:
                    cat_df[col] = 0.0
            self.categorical_normalized_df = cat_df

        # --- Option 2: Ordinal Encoding (Distinct integers 0, 1, 2...) ---
        elif method_lower == 'ordinal':
            encoder = OrdinalEncoder()
            # Fills NaNs temporarily to avoid encoder errors
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=cat_df.columns, index=cat_df.index)

        # --- Option 3: One-Hot Encoding (Binary Columns) ---
        elif method_lower == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            feature_names = encoder.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=feature_names, index=cat_df.index)

        # --- Option 4: Ordinal followed by MinMaxScaler [0, 1] ---
        elif method_lower == 'minmax_ordinal':
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            scaled_data = scaler.fit_transform(encoded_data)
            self.categorical_normalized_df = pd.DataFrame(scaled_data, columns=cat_df.columns, index=cat_df.index)

        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method='uniform')

        print(f"✨ Successfully encoded categorical data using the '{method_lower}' method.")
        return self.categorical_normalized_df     

    def create_normalized_data_df(self):
        """
        Creates a single DataFrame containing the original numeric columns 
        merged side-by-side with the normalized categorical columns.
        """
        if self.df is None: return print("Error: No data loaded.")

        # Extract numeric and normalized categorical DataFrames using existing methods
        num_df = self.extract_numeric_data()
        cat_norm_df = self.extract_normalized_categorical_data()

        # Handle the case where there are no categorical columns
        if cat_norm_df is None or (isinstance(cat_norm_df, pd.DataFrame) and cat_norm_df.empty):
            print("ℹ️ No categorical columns found. Returning numeric DataFrame only.")
            self.normalized_data_df = num_df
            return self.normalized_data_df

        # Handle the case where there are no numeric columns
        if num_df is None or (isinstance(num_df, pd.DataFrame) and num_df.empty):
            print("ℹ️ No numeric columns found. Returning normalized categorical DataFrame only.")
            self.normalized_data_df = cat_norm_df
            return self.normalized_data_df

        # Merge side-by-side along axis 1
        self.normalized_data_df = pd.concat([num_df, cat_norm_df], axis=1)
        print(f"✅ Success! Created merged DataFrame with {self.normalized_data_df.shape[1]} columns.")
        
        return self.normalized_data_df

    def plot_numerical(self, column_names):
            """
            Generates interactive Horizontal Violin, Scatter, and Histogram plots.
            Swapping the axis for Violin/Box plots to improve horizontal comparison.
            """
            if self.df is None: return
            if isinstance(column_names, str): column_names = [column_names]

            valid_cols = [c for c in column_names if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]

            for col in valid_cols:
                # Create a 1x3 grid for each numeric column
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=(f"Horizontal Violin/Box: {col}", f"Scatter Plot: {col}", f"Distribution: {col}")
                )

                # --- 1. Horizontal Violin + Box (Changed y= to x=) ---
                fig.add_trace(
                    go.Violin(x=self.df[col], box_visible=True, meanline_visible=True,
                            name=col, orientation='h', line_color='lightseagreen'),
                    row=1, col=1
                )

                # --- 2. Scatter Plot (Index vs Value) ---
                fig.add_trace(
                    go.Scatter(y=self.df[col], mode='markers',
                            marker=dict(opacity=0.5, color='royalblue'), name=col),
                    row=1, col=2
                )

                # --- 3. Histogram ---
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=col, marker_color='indianred'),
                    row=1, col=3
                )

                # Update layout for a polished look
                fig.update_layout(
                    height=450,
                    title_text=f"<b>Statistical Analysis: {col}</b>",
                    showlegend=False,
                    template="plotly_white"
                )

                # Update axes labels for clarity
                fig.update_xaxes(title_text="Value", row=1, col=1)
                fig.update_yaxes(title_text="Value", row=1, col=2)
                fig.update_xaxes(title_text="Value", row=1, col=3)

                fig.show()

    def plot_categorical(self, column_names):
        """
        Generates interactive Bar charts for categorical columns showing counts and percentages.
        """
        if self.df is None: return
        if isinstance(column_names, str): column_names = [column_names]

        for col in column_names:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(1).astype(str) + '%'

            fig = px.bar(counts, x=col, y='count', text='percentage',
                         title=f"Frequency: {col}", color=col, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.show()

    def handle_outliers(self, columns=None, find_and_delete=False):
        """
        Flags outliers using IQR logic.
        Optionally deletes the flagged rows and updates the class instance.
        """
        if self.df is None: return
        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_outliers = set()

        for col in target_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            all_outliers.update(outliers.index.tolist())
            print(f"🚨 {col}: Found {len(outliers)} outliers.")

        if all_outliers:
            display(self.df.loc[list(all_outliers)])
            if find_and_delete:
                self.df = self.df.drop(index=list(all_outliers)).reset_index(drop=True)
                print(f"🗑️ Deleted {len(all_outliers)} outlier rows.")

    def plot_relationship(self, col1, col2):
        """
        Intelligently selects the best interactive plot based on column types:
        - Num vs Num: Scatter with Trendline
        - Cat vs Num: Box plot with data points
        - Cat vs Cat: Grouped bar chart
        """
        if self.df is None: return
        is_num1, is_num2 = pd.api.types.is_numeric_dtype(self.df[col1]), pd.api.types.is_numeric_dtype(self.df[col2])

        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Correlation: {col1} vs {col2}")
        elif is_num1 != is_num2:
            num, cat = (col1, col2) if is_num1 else (col2, col1)
            fig = px.box(self.df, x=cat, y=num, points="all", color=cat, title=f"Distribution of {num} by {cat}")
        else:
            fig = px.histogram(self.df, x=col1, color=col2, barmode="group", title=f"Relationship: {col1} vs {col2}")

        fig.show()

    def plot_numerical_correlation(self):
        """
        Displays an interactive Heatmap of the Pearson Correlation matrix
        for all numeric features in the dataset.
        """
        if self.df is None: return
        
        numerical_df = self.df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                        title="Pearson Correlation Heatmap")
        fig.show()

    def plot_categorical_correlation(self):
            """
            Calculates the Cramér's V association matrix for all categorical columns
            and displays it as an interactive Plotly Heatmap.
            """
            if self.df is None: return print("Error: No data loaded.")
            
            # 1. Isolate categorical columns
            cat_df = self.df.select_dtypes(exclude=[np.number])
            
            if cat_df.empty:
                return print("⚠️ No categorical columns found to compute associations.")
                
            cols = cat_df.columns
            n_cols = len(cols)
            
            # 2. Initialize an empty matrix
            corr_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
            
            # 3. Compute pairwise Cramér's V
            for i in range(n_cols):
                for j in range(i, n_cols):
                    col1 = cols[i]
                    col2 = cols[j]
                    
                    if i == j:
                        corr_matrix.loc[col1, col2] = 1.0
                        continue
                        
                    confusion_matrix = pd.crosstab(cat_df[col1], cat_df[col2])
                    
                    if confusion_matrix.size == 0 or min(confusion_matrix.shape) <= 1:
                        corr_matrix.loc[col1, col2] = 0.0
                        corr_matrix.loc[col2, col1] = 0.0
                        continue
                        
                    chi2 = chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum().sum()
                    
                    if n > 0:
                        v = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
                    else:
                        v = 0.0
                        
                    corr_matrix.loc[col1, col2] = v
                    corr_matrix.loc[col2, col1] = v
                    
            # 4. Display the raw DataFrame
            print("--- Cramér's V Association Matrix ---")
            display(corr_matrix.round(3))
            
            # 5. Plot the interactive heatmap using Plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="<b>Cramér's V Categorical Association Heatmap</b>",
                labels=dict(color="Cramér's V")
            )
            
            # Make the plot clean and readable
            fig.update_layout(
                height=max(400, n_cols * 80),  # Dynamically scale height based on number of columns
                width=max(500, n_cols * 80),
                template="plotly_white"
            )
            
            fig.show()
            return corr_matrix

    def correlate_num_to_cat(self):
        """
        Computes associations between all numeric and categorical columns.
        - Uses Point-Biserial correlation for binary categories (-1 to 1).
        - Uses Eta (from ANOVA) for multi-class categories (0 to 1).
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns

        if len(num_cols) == 0 or len(cat_cols) == 0:
            print("⚠️ Requires both numerical and categorical columns.")
            return pd.DataFrame()

        results = []

        for cat in cat_cols:
            for num in num_cols:
                # Drop missing values for this specific pair
                valid_data = self.df[[cat, num]].dropna()
                if valid_data.empty: continue
                
                categories = valid_data[cat].unique()
                if len(categories) < 2:
                    continue  # Skip columns with only 1 unique value

                # --- Case 1: Point-Biserial Correlation for Binary categories ---
                if len(categories) == 2:
                    # Convert text categories to 0 and 1
                    binary_cat = pd.get_dummies(valid_data[cat], drop_first=True).iloc[:, 0]
                    corr, p_val = pointbiserialr(binary_cat, valid_data[num])
                    results.append({
                        'Categorical': cat,
                        'Numerical': num,
                        'Type': 'Point-Biserial (Binary)',
                        'Correlation': round(corr, 3),
                        'P-Value': round(p_val, 4)
                    })

                # --- Case 2: Eta from One-way ANOVA for Multi-class categories ---
                else:
                    groups = [valid_data[valid_data[cat] == val][num] for val in categories]
                    # Filter out any empty groups
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) > 1:
                        # Run One-Way ANOVA
                        f_val, p_val = f_oneway(*groups)
                        
                        # Calculate Eta-Squared: SS_between / SS_total
                        grand_mean = valid_data[num].mean()
                        ss_total = ((valid_data[num] - grand_mean) ** 2).sum()
                        
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        
                        if ss_total > 0:
                            eta_sq = ss_between / ss_total
                            eta = np.sqrt(eta_sq)  # Strength of the association [0 to 1]
                        else:
                            eta = 0.0

                        results.append({
                            'Categorical': cat,
                            'Numerical': num,
                            'Type': "Eta (Multi-class ANOVA)",
                            'Correlation': round(eta, 3),
                            'P-Value': round(p_val, 4)
                        })

        # Convert the results to a clean summary DataFrame
        summary_df = pd.DataFrame(results)
        return summary_df

    def plot_all_associations_heatmap(self):
            """
            Creates a unified association matrix for BOTH categorical and numeric data
            and displays it as a single interactive Plotly Heatmap.
            """
            if self.df is None: return print("Error: No data loaded.")
            
            cols = self.df.columns
            n_cols = len(cols)
            
            # 1. Initialize empty matrix
            assoc_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
            
            # 2. Iterate through every pair of columns
            for i in range(n_cols):
                for j in range(i, n_cols):
                    col1 = cols[i]
                    col2 = cols[j]
                    
                    # Diagonal is always perfectly associated
                    if i == j:
                        assoc_matrix.loc[col1, col2] = 1.0
                        continue
                    
                    # Drop rows with NaNs just for this pair
                    valid_data = self.df[[col1, col2]].dropna()
                    if valid_data.empty:
                        continue
                    
                    is_num1 = pd.api.types.is_numeric_dtype(valid_data[col1])
                    is_num2 = pd.api.types.is_numeric_dtype(valid_data[col2])
                    
                    # --- Case A: Numeric vs. Numeric (Pearson's r) ---
                    if is_num1 and is_num2:
                        val = valid_data[col1].corr(valid_data[col2], method='pearson')
                        val = abs(val)  # Absolute value to keep heatmap on a 0 to 1 scale
                        
                    # --- Case B: Categorical vs. Categorical (Cramér's V) ---
                    elif not is_num1 and not is_num2:
                        confusion_matrix = pd.crosstab(valid_data[col1], valid_data[col2])
                        if confusion_matrix.size > 0 and min(confusion_matrix.shape) > 1:
                            chi2 = chi2_contingency(confusion_matrix)[0]
                            n = confusion_matrix.sum().sum()
                            val = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1))) if n > 0 else 0.0
                        else:
                            val = 0.0
                            
                    # --- Case C: Categorical vs. Numeric (Correlation Ratio Eta) ---
                    else:
                        # Identify which is which
                        cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
                        
                        categories = valid_data[cat_col].unique()
                        if len(categories) > 1:
                            groups = [valid_data[valid_data[cat_col] == c][num_col] for c in categories]
                            groups = [g for g in groups if len(g) > 0]
                            
                            # Calculate Eta-Squared
                            grand_mean = valid_data[num_col].mean()
                            ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()
                            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                            
                            val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                        else:
                            val = 0.0
                    
                    # Assign values symmetrically
                    assoc_matrix.loc[col1, col2] = round(val, 3)
                    assoc_matrix.loc[col2, col1] = round(val, 3)
                    
            # 3. Plot the interactive Plotly Heatmap
            print("--- Global Association Matrix ---")
            display(assoc_matrix)
            
            fig = px.imshow(
                assoc_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="viridis",
                title="<b>Unified Association Heatmap (Numeric & Categorical)</b>",
                labels=dict(color="Association Strength")
            )
            
            fig.update_layout(
                height=max(500, n_cols * 45),
                width=max(600, n_cols * 45),
                template="plotly_white"
            )
            
            fig.show()
            return assoc_matrix
            
    def test_constant_mean(self, columns: Optional[Sequence[str]] = None, chunks: int = 10) -> Any:
        """
        Evaluates first moment homogeneity across sequential data blocks using MANOVA via Wilks' Lambda.
        Numerically stabilized via log-determinant tracking and shrinkage regularization.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - chunks: int, the number of non-overlapping consecutive blocks to split the data into.
        """
        if self.df is None: 
            raise ValueError("Error: No data loaded.")

        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Safety: Ensure tracking column isn't auto-selected
            if 'count' in target_cols:
                target_cols.remove('count')
            if not target_cols:
                raise ValueError("No numerical columns found automatically in the dataset.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [c for c in columns if c not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[c])]
            if non_numeric:
                raise TypeError(f"The following columns are not numerical or do not exist: {non_numeric}")
            target_cols = list(columns)

        n = len(self.df)
        m = len(target_cols)
        chunk_size = n // chunks
        if chunk_size < m:
            raise ValueError(f"Sample size per chunk ({chunk_size}) must be greater than features ({m}). Reduce chunks.")

        analysis_df = self.df[target_cols].copy()
        
        # Explicitly drop any remaining missing values row-wise to secure linear algebra operations
        analysis_df = analysis_df.dropna()
        n = len(analysis_df)
        
        analysis_df['_chunk_label'] = np.minimum(np.arange(n) // chunk_size, chunks - 1)

        # Compute Global Mean and Chunk Decompositions
        global_mean = analysis_df[target_cols].mean().values
        W = np.zeros((m, m))
        B = np.zeros((m, m))

        for label, group in analysis_df.groupby('_chunk_label'):
            X_chunk = group[target_cols].values
            chunk_mean = X_chunk.mean(axis=0)
            n_j = len(X_chunk)
            
            # Within-chunk variation
            W += np.dot((X_chunk - chunk_mean).T, (X_chunk - chunk_mean))
            # Between-chunk variation
            mean_diff = (chunk_mean - global_mean).reshape(-1, 1)
            B += n_j * np.dot(mean_diff, mean_diff.T)

        # Regularization factor to guarantee numerical stability and protect against constant sub-chunks
        epsilon = 1e-6 * np.eye(m)
        W_stable = W + epsilon
        T_stable = W + B + epsilon

        # Calculate using Stable Log-Determinants to completely bypass overflow/underflow
        sign_W, log_det_W = np.linalg.slogdet(W_stable)
        sign_T, log_det_T = np.linalg.slogdet(T_stable)

        if sign_W <= 0 or sign_T <= 0:
            raise np.linalg.LinAlgError("Variation matrices are poorly scaled or non-invertible.")

        # Wilks' Lambda derived safely in log-space: Λ = exp(log|W| - log|T|)
        log_wilks = log_det_W - log_det_T
        wilks_lambda = np.exp(log_wilks)

        # Bartlett's Chi-Square Approximation
        df_stat = m * (chunks - 1)
        scale_factor = n - 1 - (m + chunks) / 2
        chi2_calc = -scale_factor * log_wilks
        
        # Protect against edge-case negative chi2 due to tiny floating-point approximations
        chi2_calc = max(0.0, chi2_calc) 
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)

        print(f"\n--- MANOVA Mean Homogeneity Test (g={chunks} chunks, m={m} features) ---")
        print(f"Wilks' Lambda (Λ): {wilks_lambda:.5f}")
        print(f"Chi-Square Statistic: {chi2_calc:.4f} | Degrees of Freedom: {df_stat}")
        print(f"P-Value: {p_value:.6f}")
        
        if p_value > 0.05:
            print("✅ Success: Fail to reject H0. First joint moment is stable; no structural mean drift detected.")
        else:
            print("🚨 Warning: Reject H0. Significant mean drift or structural instability detected across rows.")

        return {"wilks_lambda": wilks_lambda, "chi2": chi2_calc, "p_value": p_value, "df": df_stat}

    def test_constant_covariance(self, columns: Optional[Sequence[str]] = None, chunks: int = 5) -> Any:
        """
        Evaluates second moment homogeneity across sequential data blocks using Box's M-test.
        Numerically stabilized via shrinkage regularization and row-wise imputation protection.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - chunks: int, the number of non-overlapping consecutive blocks to split the data into.
        """
        if self.df is None: 
            raise ValueError("Error: No data loaded.")

        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols:
                target_cols.remove('count')
            if not target_cols:
                raise ValueError("No numerical columns found automatically in the dataset.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [c for c in columns if c not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[c])]
            if non_numeric:
                raise TypeError(f"The following columns are not numerical or do not exist: {non_numeric}")
            target_cols = list(columns)

        analysis_df = self.df[target_cols].copy().dropna()
        n = len(analysis_df)
        m = len(target_cols)
        chunk_size = n // chunks
        
        if chunk_size <= m:
            raise ValueError(f"Degrees of freedom per chunk ({chunk_size - 1}) must be greater than number of dimensions ({m}). Reduce chunks.")

        analysis_df['_chunk_label'] = np.minimum(np.arange(n) // chunk_size, chunks - 1)
        
        S_chunks = []
        n_chunks = []
        log_det_S = 0.0
        pooled_S = np.zeros((m, m))
        total_df = 0
        
        # Stability factor matrix
        epsilon = 1e-6 * np.eye(m)
        
        for label, group in analysis_df.groupby('_chunk_label'):
            X_chunk = group[target_cols].values
            n_j = len(X_chunk)
            # Regularize individual chunk covariances to prevent singularities
            S_j = np.cov(X_chunk, rowvar=False, ddof=1) + epsilon
            
            S_chunks.append(S_j)
            n_chunks.append(n_j)
            
            df_j = n_j - 1
            pooled_S += df_j * S_j
            total_df += df_j
            
            sign, logdet = np.linalg.slogdet(S_j)
            if sign <= 0:
                raise np.linalg.LinAlgError(f"Covariance matrix for chunk {label} is non-positive definite.")
            log_det_S += df_j * logdet

        pooled_S /= total_df
        sign_p, log_det_Sp = np.linalg.slogdet(pooled_S)
        if sign_p <= 0:
            raise np.linalg.LinAlgError("Pooled covariance matrix is non-positive definite.")

        # Box's M calculation
        M = total_df * log_det_Sp - log_det_S
        
        # Scale parameter optimization factor (C)
        sum_inv_df = sum(1.0 / (nj - 1) for nj in n_chunks)
        inv_total_df = 1.0 / total_df
        numerator_C = 2.0 * m**2 + 3.0 * m - 1.0
        denominator_C = 6.0 * (m + 1.0) * (chunks - 1.0)
        C = (sum_inv_df - inv_total_df) * (numerator_C / denominator_C)
        
        chi2_calc = M * (1.0 - C)
        chi2_calc = max(0.0, chi2_calc)  # Guard against tiny negative floating-point offsets
        df_stat = (m * (m + 1) * (chunks - 1)) / 2.0
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)

        print(f"\n--- Box's M Covariance Homogeneity Test (g={chunks} chunks, m={m} features) ---")
        print(f"Box's M Statistic: {M:.4f}")
        print(f"Asymptotic Chi-Square: {chi2_calc:.4f} | Degrees of Freedom: {int(df_stat)}")
        print(f"P-Value: {p_value:.6f}")
        
        if p_value > 0.001:
            print("✅ Success: Fail to reject H0. Covariance structure is homoscedastic and stable across realizations.")
        else:
            print("🚨 Warning: Reject H0. Severe multivariate heteroscedasticity or covariance drift detected.")

        return {"M": M, "chi2": chi2_calc, "p_value": p_value, "df": int(df_stat)}

    def test_row_independence(self, columns: Optional[Sequence[str]] = None, max_lag: Optional[int] = None) -> Any:
        """
        Evaluates row-to-row statistical independence using the Multivariate Ljung-Box Portmanteau test.
        Numerically stabilized via pseudo-inverse fallbacks and row-wise missing value removal.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - max_lag: int, maximum row-index offset to analyze. Defaults to ln(n).
        """
        if self.df is None: 
            raise ValueError("Error: No data loaded.")

        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols:
                target_cols.remove('count')
            if not target_cols:
                raise ValueError("No numerical columns found automatically in the dataset.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [c for c in columns if c not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[c])]
            if non_numeric:
                raise TypeError(f"The following columns are not numerical or do not exist: {non_numeric}")
            target_cols = list(columns)

        analysis_df = self.df[target_cols].copy().dropna()
        n = len(analysis_df)
        m = len(target_cols)
        
        if max_lag is None:
            max_lag = int(np.ceil(np.log(n)))
        
        if max_lag >= n:
            raise ValueError(f"Max lag ({max_lag}) must be less than the total sample size ({n}).")

        # Center data matrix
        X = analysis_df[target_cols].values
        X_centered = X - X.mean(axis=0)
        
        # Lag 0 global variance-covariance matrix + stability shrinkage
        epsilon = 1e-6 * np.eye(m)
        Gamma_0 = (np.dot(X_centered.T, X_centered) / n) + epsilon
        
        # Safe inversion via pseudo-inverse if standard inversion is unstable
        try:
            inv_Gamma_0 = np.linalg.inv(Gamma_0)
        except np.linalg.LinAlgError:
            inv_Gamma_0 = np.linalg.pinv(Gamma_0)
        
        Q_m = 0.0
        
        # Aggregate trace variations across lags
        for k in range(1, max_lag + 1):
            # Compute cross-covariance at lag k
            Gamma_k = np.dot(X_centered[k:].T, X_centered[:-k]) / n
            
            # Trace execution: Gamma_k^T * Gamma_0^-1 * Gamma_k * Gamma_0^-1
            M_k = np.dot(np.dot(np.dot(Gamma_k.T, inv_Gamma_0), Gamma_k), inv_Gamma_0)
            trace_val = np.trace(M_k)
            
            # Update Ljung-Box inflation scalar
            Q_m += trace_val / (n - k)
            
        Q_m *= (n ** 2)
        Q_m = max(0.0, Q_m)
        df_stat = (m ** 2) * max_lag
        p_value = 1.0 - scipy.stats.chi2.cdf(Q_m, df_stat)

        print(f"\n--- Multivariate Ljung-Box Serial Independence Test (Lags Checked = {max_lag}) ---")
        print(f"Portmanteau Statistic Q_m(H): {Q_m:.4f}")
        print(f"Degrees of Freedom: {df_stat}")
        print(f"P-Value: {p_value:.6f}")
        
        if p_value > 0.05:
            print("✅ Success: Fail to reject H0. No cross-autocorrelation or row memory detected. Rows are independent.")
        else:
            print("🚨 Warning: Reject H0. Significant row-to-row serial dependency pattern identified.")

        return {"Q_m": Q_m, "p_value": p_value, "df": df_stat}

    def estimate_joint_normal(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Operationalizes the micro-scale model X_i ~ N(mu_hat, S) by fitting a 
        parametric Multivariate Normal Distribution to the verified IID baseline.
        
        Utilizes finite-sample unbiased Maximum Likelihood Estimation (MLE) with 
        Bessel's correction to construct a continuous probabilistic boundary 
        for real-time anomaly detection.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
            
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            target_cols = list(target_cols)
        else:
            target_cols = list(columns)
            
        # Extract clean baseline matrix of realizations
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        
        if n <= m:
            raise ValueError("Sample size n must be strictly larger than feature dimensions m to compute a non-singular joint covariance matrix.")
            
        # 1. Parameter Estimation 
        # mu_hat maps the spatial center of gravity of the features
        mu_hat = np.mean(X, axis=0)
        
        # S maps structural variations using ddof=1 (Bessel's correction) to correct MLE bias
        # A tiny stability shrinkage factor (epsilon) guarantees positive-definiteness
        epsilon = 1e-6 * np.eye(m)
        S_matrix = np.cov(X, rowvar=False, ddof=1) + epsilon
        
        # 2. Initialize the Continuous Scipy Distribution Object
        # Operates as the continuous Maximum Entropy boundary for raw random vectors X_i
        joint_dist = multivariate_normal(mean=mu_hat, cov=S_matrix, allow_singular=True)
        
        # 3. Compute overall model fit metrics using the joint PDF
        log_likelihoods = joint_dist.logpdf(X)
        total_log_likelihood = np.sum(log_likelihoods)
        
        # Number of parameters K = m means + (m * (m + 1) / 2) variances/covariances
        k_parameters = m + (m * (m + 1)) // 2
        aic = 2 * k_parameters - 2 * total_log_likelihood  # Akaike Information Criterion
        
        print(f"\n--- Operationalizing Micro-Scale Framework: X_i ~ N(mu_hat, S) ---")
        print(f"Dataset Scale: m={m} features, n={n} realized samples")
        print("Empirical Mean Vector (μ_hat_n):")
        for col, val in zip(target_cols, mu_hat):
            print(f"  • {col}: {val:.4f}")
        print(f"\nJoint Log-Likelihood Evaluation: {total_log_likelihood:.4f}")
        print(f"Akaike Information Criterion (AIC): {aic:.4f}")
        
        return {
            "mean_vector": mu_hat,
            "covariance_matrix": S_matrix,
            "log_likelihood": total_log_likelihood,
            "aic": aic,
            "distribution_object": joint_dist,
            "features": target_cols
        }

    def instantiate_macro_clt_distribution(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Operationalizes the macro-scale CLT model: mu_hat_n ~ N(mu_hat_n, (1/n) * S).
        
        This instantiates the continuous Gaussian sampling distribution of the empirical
        mean vector. It models parameter uncertainty rather than raw data variations,
        providing the mathematical foundation to track system drift and draw baseline 
        confidence ellipsoids.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
            
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            target_cols = list(target_cols)
        else:
            target_cols = list(columns)
            
        # Extract data to compute sample properties
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        
        if n <= m:
            raise ValueError("Sample size n must be larger than feature dimensions m to evaluate structural drift.")
            
        # 1. Compute Base Statistics
        mu_hat = np.mean(X, axis=0)
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        
        # 2. Scale Covariance to Parameter Space via the CLT Factor (1/n)
        # We inject a tiny numerical stability factor to guarantee positive-definiteness
        epsilon = 1e-10 * np.eye(m)
        clt_covariance = (1.0 / n) * S_matrix + epsilon
        
        # 3. Initialize Macro-Scale Scipy Distribution Object
        # This models the distribution of the mean vector itself
        macro_clt_dist = multivariate_normal(mean=mu_hat, cov=clt_covariance, allow_singular=True)
        
        # 4. Compute Volume / Scale Characteristics of the Uncertainty Envelope
        # Trace of the scaled matrix indicates total average squared parameter error
        total_parameter_variance = np.trace(clt_covariance)
        
        print(f"\n--- Operationalizing Macro-Scale Framework: μ_hat_n ~ N(μ_hat_n, (1/n)S) ---")
        print(f"Sample Size (n): {n} vectors collapsing parameter uncertainty")
        print("Expected Baseline Center Matrix (μ_hat_n):")
        for col, val in zip(target_cols, mu_hat):
            print(f"  • {col}: {val:.4f}")
        print(f"\nTotal Trace Error Variance (Tr[(1/n)S]): {total_parameter_variance:.8f}")
        print("Uncertainty envelope has shrunk uniformly by a factor of 1/n.")
        
        return {
            "mean_vector": mu_hat,
            "clt_covariance_matrix": clt_covariance,
            "total_parameter_variance": total_parameter_variance,
            "distribution_object": macro_clt_dist,
            "features": target_cols
        }

    def compute_empirical_pca(self, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes the geometric de-correlation framework via PCA.
        Decomposes the unbiased empirical covariance matrix S into its orthogonal 
        basis P_hat and diagonalized variance matrix Lambda_hat.
        
        Computes Hotelling's T^2 and Q (SPE) statistics across all possible 
        truncation boundaries k to optimize structural health monitoring thresholds.
        
        Generates an elite 2x3 Plotly Subplot Dashboard integrating the Feature Loading
        Heatmap alongside Eigenvalues, Variance Profiles, and T^2/Q tracking vs k.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
            
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            target_cols = list(target_cols)
        else:
            target_cols = list(columns)
            
        # Extract the matrix of realizations
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        
        if n <= m:
            raise ValueError("Sample size n must be larger than feature dimensions m to isolate principal components.")
            
        # 1. Micro-scale Centering: (x_i - mu_hat_n)
        mu_hat = np.mean(X, axis=0)
        X_centered = X - mu_hat
        
        # 2. Compute Unbiased Sample Covariance Matrix S (ddof=1)
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        
        # 3. Spectral Decomposition: S = P * Lambda * P^T
        eigenvalues, eigenvectors = np.linalg.eigh(S_matrix)
        
        # Sort in descending order to match theoretical conventions (lambda_1 >= lambda_2)
        idx = np.argsort(eigenvalues)[::-1]
        lambda_hat = eigenvalues[idx]
        P_hat = eigenvectors[:, idx]
        
        # Safeguard: Clean negative eigenvalues caused by minor floating-point errors
        lambda_hat = np.clip(lambda_hat, a_min=1e-15, a_max=None)
        
        # 4. Variance Information Calculations
        total_variance = np.sum(lambda_hat)
        if total_variance == 0:
            raise ValueError("Total variance is zero. Cannot project data into a null feature space.")
            
        explained_variance_ratio = lambda_hat / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        unexplained_variance_ratio = 1.0 - cumulative_variance_ratio
        
        # 5. Map Realizations to Principal Component Scores: z_i = P^T * (x_i - mu_hat_n)
        Z_scores = np.dot(X_centered, P_hat)
        
        # Verify de-correlation
        S_Z = np.cov(Z_scores, rowvar=False, ddof=1)
        
        # 6. Evaluate Statistical Distance Metrics (T^2 and Q) across all truncation options k
        k_range = np.arange(1, m)
        mean_T2_vs_k = []
        mean_Q_vs_k = []
        
        # Pre-allocate dictionary arrays for full sample retrieval tracking
        T2_matrix = np.zeros((n, len(k_range)))
        Q_matrix = np.zeros((n, len(k_range)))
        
        for idx_k, k in enumerate(k_range):
            # Hotelling's T^2 calculation for principal subspace: z_{i,k}^T * Lambda_k^-1 * z_{i,k}
            Z_k = Z_scores[:, :k]
            lambda_k = lambda_hat[:k]
            T2_samples = np.sum((Z_k ** 2) / lambda_k, axis=1)
            T2_matrix[:, idx_k] = T2_samples
            mean_T2_vs_k.append(np.mean(T2_samples))
            
            # Q Statistic (SPE) calculation for residual subspace: ||e_i||^2 = sum(z_{i, m-k}^2)
            Z_residual = Z_scores[:, k:]
            Q_samples = np.sum(Z_residual ** 2, axis=1)
            Q_matrix[:, idx_k] = Q_samples
            mean_Q_vs_k.append(np.mean(Q_samples))
            
        print(f"\n--- Operationalizing Geometric De-correlation Layer (PCA) ---")
        print(f"Decomposing structural space of {m} features using {n} samples.")
        print(f"Total System Variance (Trace[S]): {total_variance:.4f}")
        
        # 7. Elite 2x3 Diagnostic Subplot Dashboard
        if show_plot:
            pc_labels = [f"PC {i+1}" for i in range(m)]
            k_labels = [f"k={k}" for k in k_range]
            sensor_labels = target_cols
            
            fig = make_subplots(
                rows=2, cols=3,
                horizontal_spacing=0.18,
                vertical_spacing=0.28,
                subplot_titles=(
                    "Feature Loading Matrix |P_hat|", 
                    "Component Values (Eigenvalues λ)", 
                    "Information Profile (Explained Var.)",
                    "Residual Space (Unexplained Var.)",
                    "Mean Hotelling's T² vs Subspace Size k",
                    "Mean Q Statistic (SPE) vs Subspace Size k"
                )
            )
            
            # --- ROW 1, COL 1: PCA Loading Heatmap (Left-anchored Colorbar Scale) ---
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(P_hat), 
                    x=pc_labels, 
                    y=sensor_labels, 
                    colorscale='YlOrRd',
                    colorbar=dict(
                        title="Loading Weight", 
                        x=-0.12,          # Shifts colorbar scale safely out to the far left edge
                        len=0.38,         # Standardized length scaling matching row 1 vertical limits
                        y=0.78,           # Centers inside row 1 bounds
                        yanchor="middle",
                        xanchor="right",  # Pushes scale updates outward, protecting y-axis text strings
                        titleside="top"
                    ),
                    showscale=True, 
                    showlegend=False
                ), 
                row=1, col=1
            )
            fig.update_xaxes(title_text="Principal Axes", row=1, col=1)
            
            # --- ROW 1, COL 2: Absolute PC Eigenvalues ---
            fig.add_trace(
                go.Bar(
                    x=pc_labels, 
                    y=lambda_hat,
                    name="Eigenvalue (λ_j)",
                    marker=dict(color='#1f77b4', line=dict(color='black', width=0.5)),
                    legendgroup="eigen"
                ),
                row=1, col=2
            )
            fig.update_yaxes(title_text="Variance Magnitude", row=1, col=2)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=2)
            
            # --- ROW 1, COL 3: Information Profile (Explained Variance) ---
            fig.add_trace(
                go.Bar(
                    x=pc_labels, 
                    y=explained_variance_ratio * 100,
                    name="Marginal Explained",
                    marker=dict(color='#ff7f0e', opacity=0.75),
                    legendgroup="expl"
                ),
                row=1, col=3
            )
            fig.add_trace(
                go.Scatter(
                    x=pc_labels, 
                    y=cumulative_variance_ratio * 100,
                    mode='lines+markers',
                    name='Cumulative Captured',
                    line=dict(color='#d62728', width=2.5, dash='dash'),
                    legendgroup="expl"
                ),
                row=1, col=3
            )
            fig.update_yaxes(title_text="Captured Structure (%)", range=[-2, 105], row=1, col=3)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=3)
            
            # --- ROW 2, COL 1: Residual Unexplained Space ---
            fig.add_trace(
                go.Bar(
                    x=pc_labels, 
                    y=unexplained_variance_ratio * 100,
                    name="Remaining Noise",
                    marker=dict(color='#2ca02c', line=dict(color='black', width=0.5)),
                    legendgroup="noise"
                ),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Excluded Info (%)", range=[-2, 105], row=2, col=1)
            fig.update_xaxes(title_text="Principal Axes", row=2, col=1)
            
            # --- ROW 2, COL 2: Hotelling's T^2 vs Cutoff k ---
            fig.add_trace(
                go.Scatter(
                    x=k_labels,
                    y=mean_T2_vs_k,
                    mode='lines+markers',
                    name='Mean T²',
                    line=dict(color='#9467bd', width=2.5),
                    marker=dict(size=6, symbol='diamond'),
                    legendgroup="t2"
                ),
                row=2, col=2
            )
            fig.update_yaxes(title_text="Average T² Metric", row=2, col=2)
            fig.update_xaxes(title_text="Truncation Cutoff (k)", row=2, col=2)
            
            # --- ROW 2, COL 3: Q Statistic Residuals vs Cutoff k ---
            fig.add_trace(
                go.Scatter(
                    x=k_labels,
                    y=mean_Q_vs_k,
                    mode='lines+markers',
                    name='Mean Q (SPE)',
                    line=dict(color='#e377c2', width=2.5),
                    marker=dict(size=6, symbol='square'),
                    legendgroup="q_stat"
                ),
                row=2, col=3
            )
            fig.update_yaxes(title_text="Average Residual Energy", row=2, col=3)
            fig.update_xaxes(title_text="Truncation Cutoff (k)", row=2, col=3)
            
            # Global Unified Layout Configurations
            fig.update_layout(
                title=dict(text="Principal Component Analysis (PCA) Optimization & Feature Loading Dashboard", x=0.5, y=0.97, xanchor="center", yanchor="top"),
                template="plotly_white", 
                showlegend=True,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="center", 
                    x=0.5
                ),
                margin=dict(t=150, b=60, l=140, r=80), 
                height=750, 
                width=1250
            )
            fig.show()
            
        return {
            "mean_vector": mu_hat,
            "covariance_matrix_S": S_matrix,
            "eigenvalues_lambda": lambda_hat,
            "eigenvectors_P": P_hat,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": cumulative_variance_ratio,
            "unexplained_variance_ratio": unexplained_variance_ratio,
            "transformed_scores_Z": Z_scores,
            "score_covariance_diagonal": np.diag(S_Z),
            "features": target_cols,
            "k_values": k_range,
            "T2_matrix_vs_k": T2_matrix,
            "Q_matrix_vs_k": Q_matrix,
            "mean_T2_profile": np.array(mean_T2_vs_k),
            "mean_Q_profile": np.array(mean_Q_vs_k)
        }
        
    def compute_empirical_fa(self, k: int, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes the Factor Analysis latent subspace framework.
        Decomposes the empirical correlation matrix R into shared structural 
        variance (Lambda * Lambda^T) and individual sensor uniqueness (Psi).
        
        Estimates the latent common factor scores using Thomson's MMSE regression method.
        
        Generates an elite 4-panel Plotly Subplot Dashboard analyzing Factor Loadings,
        Communality vs Uniqueness balances, Sensor Noise Scales, and Latent Factor Energy.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
            
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'count' in target_cols: target_cols.remove('count')
            target_cols = list(target_cols)
        else:
            target_cols = list(columns)
            
        # Extract data matrix of realizations
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        
        if n <= m:
            raise ValueError("Sample size n must be larger than feature dimensions m to isolate latent factors.")
        if k >= m:
            raise ValueError(f"Latent factors k ({k}) must be strictly less than visible features m ({m}).")
            
        # 1. Standardize realizations to construct the Z space: Z = D^(-1/2) * (X - mu_hat)
        mu_hat = np.mean(X, axis=0)
        std_hat = np.std(X, axis=0, ddof=1)
        
        # Safeguard against zero variance
        std_hat[std_hat == 0] = 1e-15
        Z = (X - mu_hat) / std_hat
        
        # Compute the empirical correlation matrix R
        R_matrix = np.corrcoef(X, rowvar=False)
        
        # 2. Fit the Factor Analysis Model via Maximum Likelihood / SVD
        fa = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
        fa.fit(Z)
        
        # Extract core FA parameters
        # Lambda matrix: (m x k)
        lambda_matrix = fa.components_.T 
        # Psi diagonal vector: (m,)
        psi_diagonal = fa.noise_variance_ 
        
        # 3. Calculate Communality (h^2) and Uniqueness (phi^2) vectors
        communality = np.sum(lambda_matrix**2, axis=1)
        uniqueness = psi_diagonal  # mathematically equal under the model
        
        # 4. Map realizations to Latent Space using Thomson's Regression Method:
        # F_scores = Z * R^(-1) * Lambda
        F_scores = fa.transform(Z)
        
        print(f"\n--- Operationalizing Latent Subspace Layer (Factor Analysis) ---")
        print(f"Decomposing structural space of {m} sensors into {k} hidden physical factors.")
        print(f"Average System Communality (Shared Subspace Energy): {np.mean(communality)*100:.2f}%")
        print(f"Average System Uniqueness (Localized Noise Floor): {np.mean(uniqueness)*100:.2f}%")
        
        # 5. Elite 4-panel Horizontal Diagnostics Subplot Dashboard
        if show_plot:
            sensor_labels = target_cols
            factor_labels = [f"Factor {j+1}" for j in range(k)]
            
            fig = make_subplots(
                rows=2, cols=2,
                horizontal_spacing=0.24,  # Padded to leave breathing room for sensor labels
                vertical_spacing=0.28,    # Prevents x-axis text strings from clipping titles
                subplot_titles=(
                    "Structural Loadings Matrix |λ_(j,r)|", 
                    "Variance Partitioning (Communality vs Uniqueness)", 
                    "Sensor Uniqueness Noise Floor (φ²)",
                    "Latent Factor Scores Empirical Variance"
                )
            )
            
            # --- Subplot 1: Factor Loadings Heatmap (Left-anchored scale matching PCA layout) ---
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(lambda_matrix),
                    x=factor_labels,
                    y=sensor_labels,
                    colorscale='YlOrRd',
                    colorbar=dict(
                        title="Sensitivity Score", 
                        x=-0.15,          # Perfectly isolates colorbar to the left side 
                        len=0.38,         # Fits height bounds of row 1 perfectly
                        y=0.78,           # Centers vertically inside row 1 bounds
                        yanchor="middle",
                        xanchor="right",  # Protects y-axis text tags from getting crushed
                        titleside="top"
                    ),
                    showscale=True,
                    name="Loadings"
                ),
                row=1, col=1
            )
            fig.update_xaxes(title_text="Latent Structures", row=1, col=1)
            
            # --- Subplot 2: Communality vs Uniqueness Stacked Bar ---
            fig.add_trace(
                go.Bar(
                    y=sensor_labels, x=communality * 100,
                    name="Communality (h² - Shared Structure)",
                    orientation='h',
                    marker=dict(color='#1f77b4')
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(
                    y=sensor_labels, x=uniqueness * 100,
                    name="Uniqueness (φ² - Channel Noise)",
                    orientation='h',
                    marker=dict(color='#ff7f0e')
                ),
                row=1, col=2
            )
            fig.update_layout(barmode='stack')
            fig.update_xaxes(title_text="Variance Allocation (%)", range=[0, 100], row=1, col=2)
            
            # --- Subplot 3: Specific Sensor Uniqueness Line ---
            fig.add_trace(
                go.Scatter(
                    x=sensor_labels, y=uniqueness,
                    mode='lines+markers',
                    name='Uniqueness Profile (φ²)',
                    line=dict(color='#d62728', width=2, dash='dot'),
                    marker=dict(size=8, symbol='x')
                ),
                row=2, col=1
            )
            fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
            fig.update_xaxes(title_text="Monitored Channels", tickangle=25, row=2, col=1)
            
            # --- Subplot 4: Variance of Factor Scores ---
            factor_variances = np.var(F_scores, axis=0, ddof=1)
            fig.add_trace(
                go.Bar(
                    x=factor_labels, y=factor_variances,
                    name="Factor Empirical Variance",
                    marker=dict(color='#2ca02c', line=dict(color='black', width=0.5))
                ),
                row=2, col=2
            )
            fig.update_yaxes(title_text="Variance Level", row=2, col=2)
            fig.update_xaxes(title_text="Latent Vectors", row=2, col=2)
            
            # Global Unified Layout Configurations
            fig.update_layout(
                title=dict(
                    text="Factor Analysis (FA) Latent Subspace Diagnostics Dashboard",
                    x=0.5, y=0.97,
                    xanchor="center", yanchor="top"
                ),
                template="plotly_white",
                showlegend=True,
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", y=1.02, 
                    xanchor="center", x=0.5
                ),
                margin=dict(t=150, b=60, l=140, r=80),
                height=750,
                width=1250
            )
            fig.show()
            
        return {
            "mean_vector_mu": mu_hat,
            "std_vector_D": std_hat,
            "correlation_matrix_R": R_matrix,
            "factor_loadings_lambda": lambda_matrix,
            "uniqueness_psi": uniqueness,
            "communality_h2": communality,
            "latent_factor_scores_F": F_scores,
            "sensors": target_cols
        }
                
import time
from datetime import datetime
import uuid

import json
import pandas as pd
import copy
import inspect
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import networkx as nx

import graphviz
import base64
from IPython.display import HTML


class PlottingMethods:

    def get_methods_info(self, user_id=None):
        method_dicts=[]
        # Introspect class methods
        methods = inspect.getmembers(self, inspect.ismethod)

        for name, method in methods:
            # Skip private methods
            if name.startswith('_'):
                continue

            # Get method signature
            signature = inspect.signature(method)

            # Get docstring
            docstring = method.__doc__
            formatted_docstring = docstring.strip() if docstring else "No description available"

            # Append method documentation
            method_dicts+= [{"method": name, "signature":str(signature), "description": formatted_docstring}]
        status='success'
        return {'status':status,'response':method_dicts}


    def _data_validate(self, data, message_dict):
            # 1. Handle empty/None input
            if data is None or (isinstance(data, str) and not data):
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}

            # 2. Check if data is a DataFrame
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    message_dict.update({'message': 'No data'})
                    return {'status': 'error', 'message_dict': message_dict}
                return {'status': 'success', 'data': data.to_dict(orient='records')}

            # 3. Check if data is a list (records)
            if isinstance(data, list):
                if not data:
                    message_dict.update({'message': 'No data'})
                    return {'status': 'error', 'message_dict': message_dict}
                return {'status': 'success', 'data': data}

            # 4. Handle JSON string input
            try:
                parsed_data = json.loads(data)
                # Support both {'records': [...]} structure OR raw list of records
                records = parsed_data.get('records') if isinstance(parsed_data, dict) else parsed_data

                if not records:
                    message_dict.update({'message': 'No data'})
                    return {'status': 'error', 'message_dict': message_dict}

                return {'status': 'success', 'data': records}

            except (json.JSONDecodeError, TypeError):
                message_dict.update({'message': 'Invalid data format'})
                return {'status': 'error', 'message_dict': message_dict}


    def plot_bar_chart(self, x='date', y='value', color=None, text=None, title='', barmode='stack', hover_data=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px bar chart with x as the x values,
        y as the y values, and color as the categories. The variable barmode specifies if the mode of grouping.

        Args:
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            color (str): Optional - Column name for the stacking categories.
            text (str or None): Optional - Column name for text labels.
            title (str): Optional - Title of the chart.
            barmode (str or None): Optional - The bar mode either stack or group.
            hover_data (list or None): Optional - List of column names to include in hover data.
            data (str): JSON string containing a list of records in the format
                        {'records': [{'x': ..., 'y': ..., 'color': ...}, ...]}.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """



        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            if isinstance(hover_data, str):
                try:
                    # First, attempt to parse it as JSON
                    parsed_data = json.loads(hover_data)

                    # If parsing succeeds and the result is a list, use it
                    if isinstance(parsed_data, list):
                        hover_data = parsed_data
                    else:
                        # If it's valid JSON but not a list, set hover_data to None or handle as needed
                        hover_data = None
                except json.JSONDecodeError:
                    # If JSON parsing fails, check if it's a comma-separated string
                    if ',' in hover_data:
                        hover_data = hover_data.split(',')
                    else:
                        # Handle non-JSON, non-comma-separated strings
                        hover_data = None

            include_plotlyjs=True
            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[y] = pd.to_numeric(df[y])
            c_categories_labels = None
            if color is not None:
                df.dropna(subset=[color], inplace=True)
                # Use unique() and sort if necessary
                c_categories_labels = df[color].unique()
                if not any(sub in color.lower() for sub in ['month', 'week']):
                    c_categories_labels = sorted(c_categories_labels)
                df[color] = pd.Categorical(df[color], categories=c_categories_labels, ordered=True)

            # Ensure xLabel sorting
            x_categories_labels = df[x].unique()
            df[x] = pd.Categorical(df[x], categories=x_categories_labels, ordered=True)

            if hover_data:
                hover_data = [col for col in hover_data if col in df.columns]


            # Generate a stacked bar graph
            cat_orders = {x: x_categories_labels}

            # 2. Only add color to category_orders if color is actually provided
            if color is not None and c_categories_labels is not None:
                cat_orders[color] = c_categories_labels

            # 3. Generate a stacked bar graph using the dynamic dictionary
            fig = px.bar(
                df,
                x=x,
                y=y,
                color=color,
                title=title,
                text=text,
                hover_data=hover_data,
                category_orders=cat_orders  # Use the cleaned dictionary
            )

            # Customize the layout (optional)
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y,
                uniformtext_minsize=8,
                uniformtext_mode='hide',  # Hide overlapping text
                barmode=barmode           # Enable stacked mode
            )


            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the response with the figure
            message_dict.update({'message': 'Bar chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        return response

    def plot_pie_chart(self, names='date', values='value', title='', hole=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates a responsive Plotly pie chart based on provided or previously stored data.

        This method creates a pie chart using the Plotly Express library. Data can be provided directly
        via a JSON string (`data`) or indirectly through a reference ID (`data_id`) which is used to
        retrieve previously stored data.

        Parameters:
            names (str): Column name in the dataset to use for pie chart segment labels (default: 'date').
            values (str): Column name in the dataset to use for segment sizes (default: 'value').
            title (str): Title of the pie chart (default: empty string).
            data_id (str, optional): Optional ID for retrieving stored data via `DBQ.get_ai_message_stored_data`.
            data (str, optional): JSON string in the format `{"records": [...]}` representing the dataset.
                                Used if `data_id` is not provided.
            meta_data (dict): Additional metadata dictionary to include in the response.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if the chart is generated; 'error' if there is an issue.
                - 'response' (dict):
                    - 'meta_data' (dict): Includes original meta plus status or error message.
                    - 'data' (str): JSON-encoded string containing the HTML representation of the pie chart.
                    - 'message' (str): A stringified version of `meta_data` for convenience.

        Example:
            >>> plot_pie_chart(
                    names='category',
                    values='count',
                    title='Category Distribution',
                    data=json.dumps({"records": [{"category": "A", "count": 40}, {"category": "B", "count": 60}]})
                )
            {
                'status': 'success',
                'response': {
                    'meta_data': {'message': 'Pie chart plotted'},
                    'data': '{"figure": "<div id='abc123'>...</div>"}',
                    'message': '{"message": "Pie chart plotted"}'
                }
            }

        Notes:
            - The generated Plotly chart is converted into HTML for embedding in web pages.
            - If both `data` and `data_id` are missing or invalid, an appropriate error is returned.
            - Ensures responsiveness and disables the Plotly logo in the exported chart.
            - Each chart is given a unique `div` ID for safe embedding.

        Raises:
            Exception: Any errors during data retrieval or plotting are caught and returned in the response.
        """


        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            include_plotlyjs=True
            # Load data into a pandas DataFrame
            df = pd.DataFrame(data)

            # Generate a pie chart
            fig = px.pie(
                df,
                names=names,  # Category labels
                values=values,  # Values corresponding to each category
                title=title,
                hole=hole
            )

            # Customize the layout (optional)
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(
                title=title,
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Pie chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}
        return response

    def plot_histogram(self, x='value', title='', bins=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px histogram with the specified column as the x-axis.

        Args:
            x (str): Column name for the x-axis.
            title (str): Title of the histogram.
            bins (list or None): Custom bin intervals for the histogram.
            data (str): JSON string containing a list of records.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """

        message_dict={'message':meta_data}
        validated_response=self._data_validate(data, message_dict)

        if not validated_response.get('status')=='success':
            message_dict=validated_response.get('message_dict',{})
            message=message_dict.get('message','Error passing data')
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
        data=validated_response.get('data')


        include_plotlyjs=True

        # Load data into a pandas DataFrame
        df = pd.DataFrame(data)

        # If custom bins are provided, preprocess the data
        if bins:
            if not isinstance(bins, list) or len(bins) < 2:
                return {
                    'status': 'error',
                    'response': {
                        'meta_data': 'Invalid bins parameter. Must be a list with at least two elements.',
                        'data': {'figure': ''}
                    }
                }
            # Use pandas.cut to bin the data
            df[x] = pd.cut(df[x], bins=bins, right=False)
            # Convert intervals to strings
            df[x] = df[x].astype(str)

        # Generate a histogram
        fig = px.histogram(
            df,
            x=x,        # Data for x-axis
            title=title,
            category_orders={x: [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]} if bins else None
        )

        # Customize the layout (optional)
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title='Count',
            bargap=0.2  # Adjust gap between bars
        )

        plot_html = pio.to_html(
            fig,
            full_html=False,
            config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
            include_plotlyjs=include_plotlyjs #'cdn'
        )
        fig_id = str(uuid.uuid4())[:8]
        fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

        message_dict.update({'message': 'Histogram plotted'})
        message = json.dumps(message_dict)
        response = {'status': 'success', 'response': {'meta_data': message, 'data': json.dumps({'figure': fig_return})}}
        return response

    def plot_simple_sunburst_graph(self, path=["parent", "name"], values="marks", title='Hierarchy map', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plots a Plotly Sunburst graph given hierarchical data.

        Args:
            data (str): JSON string of hierarchical records.
            path (list): List defining the hierarchy in the dataset.
            values (str): Column name to use for slice sizes.
            title (str): Title of the Sunburst plot.

        Returns:
            dict: A response containing the Plotly figure.
        """

        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            include_plotlyjs=True

            # Parse the JSON string into a DataFrame
            df = pd.DataFrame(data)
            df = df.fillna('')

            # Generate the Sunburst figure
            fig = px.sunburst(
                df,
                path=path,
                values=values,
                title=title
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Pie chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message, 'data': json.dumps({'figure': fig_return})}}
        except Exception as e:
            response = {'status': 'error', 'response': {'meta_data': json.dumps({'data_id':data_id,'message':f'Error: {str(e)}'}), 'data': json.dumps({'figure': ''})}}


        return response

    def plot_tree_map(self, path=["parent", "name"], values="marks", title='Hierarchy map', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        '''
        Given a list of hierarchical dictionaries, plot a plotly px treemap with the specified path.
        '''

        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            include_plotlyjs=True
            df = pd.DataFrame(data)

            fig = px.treemap(
                df,
                path=path,
                values=values,
                title=title
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Plot tree plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message, 'data': json.dumps({'figure': fig_return})}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message, 'data': json.dumps({'figure': ''})}}

        return response

    def plot_sankey_diagram(self, source_column="parent", target_column="name", values="marks", title="Sankey Diagram", data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plot a Plotly Sankey diagram
        using the specified source and target columns.

        Args:
            source_column (str): The column name to use as the source in the Sankey diagram.
            target_column (str): The column name to use as the target in the Sankey diagram.
            values (str): The column name to use as the values in the Sankey diagram.
            title (str): The title of the Sankey diagram.
            data (str): JSON string containing the records of hierarchical data.

        Returns:
            dict: A dictionary containing:
                  - status (str): 'success' or 'error'.
                  - response (dict):
                      - message (str): Success or error message.
                      - data (dict): Contains the Sankey diagram figure as a Plotly figure object.
        """


        include_plotlyjs=True
        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            # Convert the input data to a DataFrame
            df = pd.DataFrame(data)
            grouped_df = df.groupby([source_column, target_column], as_index=False).agg({values: 'sum'})
            # Create lists for Sankey diagram nodes and links
            unique_nodes = pd.concat([grouped_df[source_column], grouped_df[target_column]]).unique()
            node_map = {node: idx for idx, node in enumerate(unique_nodes)}  # Map nodes to indices

            # Prepare the Sankey diagram's links
            sources = grouped_df[source_column].map(node_map).tolist()
            targets = grouped_df[target_column].map(node_map).tolist()
            values = grouped_df[values].tolist()

            # Create the Sankey diagram using Plotly
            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=list(node_map.keys()),  # Node labels
                        ),
                        link=dict(
                            source=sources,  # Source nodes
                            target=targets,  # Target nodes
                            value=values,    # Values (weights)
                        ),
                    )
                ]
            )

            # Add title to the figure
            fig.update_layout(title_text=title, font_size=10)

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Sankey diagram plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message, 'data': json.dumps({'figure': fig_return})}}
            return response

        except Exception as e:
            # Handle unexpected errors
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}
            return response

    def plot_sunburst_from_hierarchy(self, path=["parent", "name"], values="marks", title="Sunburst Diagram",data_id=None, data="{'records':[]}", meta_data={}, user_id=None):
        """
        Generate a Plotly Sunburst chart from hierarchical data.
        """
        message_dict={'message':meta_data}
        validated_response=self._data_validate(data, message_dict)

        if not validated_response.get('status')=='success':
            message_dict=validated_response.get('message_dict',{})
            message=message_dict.get('message','Error passing data')
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
        data=validated_response.get('data')

        include_plotlyjs=True

        def minimal_sunburst_hierarchy(
            data: str,
            path: List[str] = ["parent", "name"],
            values: str = "marks",
            top_level: Optional[str] = None,
            default_root: str = "Root"
        ) -> pd.DataFrame:
            """
            Build a hierarchical DataFrame for a Sunburst chart with aggregated marks.

            Args:
                data (str): JSON string containing hierarchical records.
                path (list): Columns specifying parent-child relationships.
                values (str): Column specifying values (marks) for the chart.
                top_level (str or None): Name of the top-level node in the hierarchy. If None, it will be inferred.
                default_root (str): Default root node name if multiple top-level nodes are detected.

            Returns:
                pd.DataFrame: Hierarchical DataFrame with aggregated marks for all nodes.
            """
            # Parse input data
            try:
                records = json.loads(data).get("records", [])
                if not records:
                    raise ValueError("Input data is empty or invalid.")
                df = pd.DataFrame(records)
            except Exception as e:
                raise ValueError(f"Failed to parse data: {e}")

            # Clean and prepare the DataFrame
            df[path[0]] = df[path[0]].replace([None, '', pd.NA], default_root).str.strip()
            df[path[1]] = df[path[1]].str.strip()

            # Infer the top-level node if not provided
            if top_level is None:
                potential_top_levels = set(df[path[0]].unique()) - set(df[path[1]].unique())
                top_level = potential_top_levels.pop() if len(potential_top_levels) == 1 else default_root

            # Ensure the top-level node exists in the DataFrame
            if top_level not in df[path[0]].values:
                df = pd.concat([
                    pd.DataFrame({path[0]: [default_root], path[1]: [top_level], values: [None]}),
                    df
                ], ignore_index=True)

            # Build the hierarchy dictionary
            def build_hierarchy(df: pd.DataFrame, path: List[str]) -> Dict[str, List[str]]:
                hierarchy = {}
                for _, row in df.iterrows():
                    parent, child = row[path[0]], row[path[1]]
                    hierarchy.setdefault(parent, []).append(child)
                return hierarchy

            hierarchy_dict = build_hierarchy(df, path)

            # Aggregate marks for intermediate levels
            def aggregate_marks(node: str, hierarchy: Dict[str, List[str]], values_dict: Dict[str, float]) -> float:
                if node in values_dict:
                    return values_dict[node]
                if node in hierarchy:
                    child_sum = sum(aggregate_marks(child, hierarchy, values_dict) for child in hierarchy[node])
                    values_dict[node] = child_sum
                    return child_sum
                return 0

            values_dict = {row[path[1]]: row[values] for _, row in df.iterrows() if pd.notnull(row[values])}
            for node in hierarchy_dict:
                aggregate_marks(node, hierarchy_dict, values_dict)

            # Expand the hierarchy into paths
            def expand_hierarchy(node: str, hierarchy: Dict[str, List[str]], current_path: List[str]) -> List[List[str]]:
                current_path.append(node)
                if node in hierarchy:
                    return [path for child in hierarchy[node] for path in expand_hierarchy(child, hierarchy, current_path.copy())]
                return [current_path]

            all_paths = expand_hierarchy(top_level, hierarchy_dict, [])

            # Construct the hierarchical DataFrame
            max_depth = max(len(path) for path in all_paths)
            hierarchical_data = [
                {f"level_{i}": (path[i] if i < len(path) else None) for i in range(max_depth)} | {"marks": values_dict[path[-1]]}
                for path in all_paths
            ]

            return pd.DataFrame(hierarchical_data)

        # Parse and construct the hierarchy
        hierarchy_results = minimal_sunburst_hierarchy(data)
        hierarchical_df = hierarchy_results

        # Determine path columns for the Sunburst chart
        path_columns = [col for col in hierarchical_df.columns if hierarchical_df[col].notnull().any()]

        # Generate the Sunburst chart
        fig = px.sunburst(
            hierarchical_df,
            path=path_columns,
            values=values,  # Currently, marks are not mapped; can be added if needed.
            title=title
        )

        plot_html = pio.to_html(
            fig,
            full_html=False,
            config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
            include_plotlyjs=include_plotlyjs #'cdn'
        )
        fig_id = str(uuid.uuid4())[:8]
        fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

        message_dict.update({'message': 'Pie chart plotted'})
        message = json.dumps(message_dict)
        response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}
        return response

    def plot_heat_map(self,  values='Sales', index='Region', columns='Category', aggregade_method='sum', fill_value=0,  title='Heatmap of Normalized Marks', width=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates an interactive Plotly heatmap from tabular data, with optional aggregation and layout control.

        This method builds a heatmap visualization using a pivoted version of the input dataset.
        Data can be provided either directly (`data`) or fetched via a reference `data_id` using a stored data lookup.

        Parameters:
            values (str): The column whose values will be visualized in the heatmap (e.g., grades, sales).
            index (str): The column to use as the y-axis in the pivot table.
            columns (str): The column to use as the x-axis in the pivot table.
            aggregade_method (str): Aggregation method to apply when pivoting ('sum', 'mean', 'count', etc.). Defaults to 'sum'.
            fill_value (int or float): Value used to fill missing cells in the pivot table. Defaults to 0.
            title (str): Title of the heatmap.
            width (int, optional): Optional fixed width of the chart in pixels. If None, it's auto-sized.
            data_id (str, optional): Optional ID to fetch data from stored source using `DBQ.get_ai_message_stored_data`.
            data (str, optional): A JSON string in the format `{"records": [...]}`. Used if `data_id` is not provided.
            meta_data (dict): Metadata dictionary to propagate through the response.

        Returns:
            dict: A dictionary containing:
                - 'status' (str): 'success' if the heatmap was created, otherwise 'error'.
                - 'response' (dict):
                    - 'meta_data' (dict): Extended metadata including any error or success messages.
                    - 'data' (str): JSON string with an HTML-embedded Plotly figure (`figure`).
                    - 'message' (str): Same as meta_data['message'], serialized for frontend use.

        Example:
            >>> plot_heat_map(
                    values='grade',
                    index='student_name',
                    columns='module',
                    aggregade_method='mean',
                    title='Average Grades per Module',
                    data=json.dumps({"records": [{"student_name": "Alice", "module": "Math", "grade": 85}, ...]})
                )
            {
                'status': 'success',
                'response': {
                    'meta_data': {'message': 'Heat map plotted'},
                    'data': '{"figure": "<div id='abcd1234'>...</div>"}',
                    'message': '{"message": "Heat map plotted"}'
                }
            }

        Notes:
            - The pivoted DataFrame is plotted using `plotly.express.imshow()`.
            - The chart is rendered in HTML for easy embedding in web pages or dashboards.
            - Unique div IDs are assigned to avoid DOM conflicts in the frontend.
            - The layout is responsive and resizes fluidly within containers.

        Raises:
            Exception: Any errors during data processing or plotting are captured and returned in the response.
        """

        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            include_plotlyjs=True
            df = pd.DataFrame(data)
            col_labels = df[columns].unique()
            row_labels = df[index].unique()

            pivot_data = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggregade_method, fill_value=0)
            # Reindex to original order
            pivot_data = pivot_data.reindex(index=row_labels, columns=col_labels)
            transposed_pivot_data = pivot_data.T

            # Create heatmap using plotly express
            fig = px.imshow(
                pivot_data, # transposed_pivot_data,  # DataFrame without 'X_Labels' column
                color_continuous_scale='Jet', #'Viridis',  # Choose any colorscale
                labels=dict(y=index, x=columns, color=values),
                text_auto=True
            )

            # Add title and make layout responsive
            fig.update_layout(
                title=title,
                autosize=True,  # Automatically adjust figure size
                template='plotly',  # Optional: set a clean template
            )

            # Enable responsive resizing
            fig.update_layout(
                width=width,  # Let the browser handle width
                height=None,  # Optional: make height dynamic as well
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Heat map plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}
        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}

        return response

    def plot_multi_column_bar_graph(
        self,
        xLabel="Week",
        value_vars=['Total Slots Allocated by Region', 'Containers Product Ordered by Region'],
        title="Slot Allocation",
        hover_data=[],
        barmode='group',
        data_id=None,
        data='{"records":[]}',
        meta_data={},
        user_id=None
    ):

        include_plotlyjs = True


        try:
            message_dict={'message':meta_data}
            validated_response=self._data_validate(data, message_dict)

            if not validated_response.get('status')=='success':
                message_dict=validated_response.get('message_dict',{})
                message=message_dict.get('message','Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}
            data=validated_response.get('data')


            parsed_data = data
            records = parsed_data.get('records', [])
            if not records:
                message_dict.update({'message': 'No data'})
                message=json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}

        except Exception:
            message_dict.update({'message': 'Invalid or missing data'})
            message=json.dumps(message_dict)
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}

        df = pd.DataFrame(records)
        needed_columns = list(set([xLabel] + hover_data + value_vars))
        df = df[needed_columns]

        df_melted = df.melt(
            id_vars=[xLabel] + hover_data,
            value_vars=value_vars,
            var_name='Group',
            value_name='Value'
        )

        fig = px.bar(
            df_melted,
            x=xLabel if orientation == 'v' else 'Value',
            y='Value' if orientation == 'v' else xLabel,
            color='Group',
            barmode=barmode,
            title=title,
            hover_data=hover_data if hover_data else None,
            orientation='v'
        )

        fig.update_layout(
            title_font=dict(size=12),
            title_automargin=True,
            title_pad=dict(t=5),
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(
                font=dict(size=10),
                orientation="h",
                x=0.5, xanchor="center",
                y=1, yanchor="bottom"
            ),
            autosize=True,
            font=dict(size=9)
        )

        fig_return = pio.to_html(
            fig, full_html=False, include_plotlyjs=include_plotlyjs, config={"displaylogo": False, "responsive": True}
        )

        message_dict.update({'message': 'Bar chart plotted'})
        message = json.dumps(message_dict)
        response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}
        return response


    def plot_flow_chart(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns a flowchart visualization from given process data using Graphviz and NetworkX.

        This method accepts either a data_id to fetch pre-stored data, or a raw data string containing flowchart
        records in JSON format. The resulting flowchart is rendered as a PNG image, base64-encoded, and returned
        as HTML along with meta information and status.

        Edge and node attributes are accessed using `.get()` with sensible defaults, ensuring robust handling of missing fields.
        Edges now support optional custom color and thickness (penwidth).

        Args:
            data_id (str, optional): An identifier used to retrieve stored process data. If provided, overrides `data`.
            data (str, optional): A JSON string with a "records" dictionary, containing "edges" and "nodes" for the flowchart.
            meta_data (dict, optional): Metadata dictionary to include in the response.
            user_id (str, optional): Optional user identifier for logging or auditing purposes.

        Returns:
            dict: A dictionary containing:
                - 'status': 'success' or 'error'
                - 'response': {
                    'meta_data': Meta information and messages,
                    'data': JSON string with an embedded PNG image (base64-encoded HTML figure)
                }
                - 'message': JSON-encoded message describing the outcome.

        Raises:
            None: All exceptions are caught and reported in the 'status':'error' response.

        Data Structure:
            The input data should be a JSON object with the following format:

            {
                "records": {
                    "edges": [
                        {
                            "start": "<source_node_label>",      # (str) Required. The source node's label or ID.
                            "end": "<destination_node_label>",   # (str) Required. The destination node's label or ID.
                            "label": "<edge_label>",             # (str) Optional. The label displayed on the edge.
                            "color": "<color>",                  # (str) Optional. Edge color (e.g. '#ff0000', 'blue'). Default: 'black'
                            "penwidth": <number>,                # (float/int) Optional. Edge thickness. Default: 1
                        },
                        ...
                    ],
                    "nodes": [
                        {
                            "label": "<node_label>",         # (str) Required. Node's unique label or ID.
                            "shape": "<shape>",              # (str) Optional. Graphviz shape (e.g. 'box', 'ellipse'). Default: 'ellipse'
                            "style": "<style>",              # (str) Optional. Node style (e.g. 'filled'). Default: 'filled'
                            "fillcolor": "<color>",          # (str) Optional. Fill color (e.g. '#abcdef'). Default: '#bbbbbb'
                            "fontcolor": "<color>"           # (str) Optional. Font color (e.g. '#000000'). Default: 'black'
                        },
                        ...
                    ]
                }
            }

        Example:
            result = self.plot_flow_chart(data_id='abc123')
            # or
            result = self.plot_flow_chart(data='{"records": {"edges": [...], "nodes": [...]}}')

        Notes:
            - At least one of `data_id` or `data` must provide valid flowchart records.
            - Handles missing or invalid data gracefully, always returning a well-structured response.
            - Uses `.get()` with default values for node and edge attributes to prevent errors from missing fields.
            - Edges can be customized with `color` and `penwidth`.
            - Uses Graphviz to render the flowchart, and NetworkX for graph data manipulation.

        """

        dot = graphviz.Digraph(format='png')
        G = nx.MultiDiGraph()

        message_dict={'message':meta_data}
        if not data and not data_id:
            message_dict.update({'message': 'No data or data_id'})
            message = json.dumps(message_dict)
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        if data_id:
            try:
                data_retrieval_response = DBQ.get_ai_message_stored_data(data_id)
                data=data_retrieval_response['response']['data']
            except Exception as e:
                message_dict.update({'message': 'No data'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        if not data_id and data:
            if not json.loads(data).get('records',None):
                message_dict.update({'message': 'No data'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        records = json.loads(data).get('records', [])
        states = records.get('edges',[])
        node_properties = records.get('nodes',[])

        try:
            # Add edges with attributes
            for state in states:
                edge_label = f"{state.get('label','')}"
                edge_color = state.get('color', 'black')       # <-- default to black if missing
                edge_penwidth = str(state.get('penwidth', 1))  # <-- default to 1 if missing

                dot.edge(
                    state.get('start', ''),
                    state.get('end', ''),
                    label=edge_label,
                    color=edge_color,
                    penwidth=edge_penwidth
                )
                G.add_edge(
                    state.get('start', '').split(':')[0],
                    state.get('end', '').split(':')[0],
                    label=edge_label,
                    fontcolor='black',
                    color=edge_color,
                    penwidth=edge_penwidth
                )

            # Add node properties
            for node_p in node_properties:
                label = node_p.get('label', '')
                if label in list(G.nodes()):
                    shape = node_p.get('shape', 'ellipse')
                    style = node_p.get('style', 'filled')
                    fillcolor = node_p.get('fillcolor', '#bbbbbb')
                    fontcolor = node_p.get('fontcolor', 'black')

                    G.add_node(
                        label,
                        shape=shape,
                        style=style,
                        fillcolor=fillcolor,
                        fontcolor=fontcolor
                    )
                    dot.node(
                        label,
                        shape=shape,
                        style=style,
                        fillcolor=fillcolor,
                        color=fontcolor
                    )

            # Render the graph directly to a byte stream (PNG)
            png_bytes = dot.pipe(format='png')
            encoded = base64.b64encode(png_bytes).decode("utf-8")

            # Create HTML for the image
            plot_html = f"""
            <div>
            <figure>
                <img src="data:image/png;base64,{encoded}" alt="Flowchart">
            </figure>
            </div>
            """

            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the figure in the response
            message_dict.update({'message': 'Flow chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message':message}}


        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}

        return response

    def plot_flow_chart_plotly(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns a flowchart visualization from given process data using Plotly and NetworkX.
        (Docstring omitted for brevity - keep yours.)
        """
        message_dict={'message':meta_data}
        if not data and not data_id:
            message_dict.update({'message': 'No data or data_id'})
            message = json.dumps(message_dict)
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        if data_id:
            try:
                data_retrieval_response = DBQ.get_ai_message_stored_data(data_id)
                data=data_retrieval_response['response']['data']
            except Exception as e:
                message_dict.update({'message': 'No data'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        if not data_id and data:
            if not json.loads(data).get('records',None):
                message_dict.update({'message': 'No data'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message':message}

        records = json.loads(data).get('records', {})
        states = records.get('edges',[])
        node_properties = records.get('nodes',[])

        try:
            # Build the NetworkX graph
            G = nx.MultiDiGraph()
            edge_labels = {}
            for state in states:
                start = state['start'].split(':')[0]
                end = state['end'].split(':')[0]
                label = state.get('label', '')
                G.add_edge(start, end)
                edge_labels[(start, end)] = label

            for node_p in node_properties:
                if node_p['label'] in G.nodes:
                    G.nodes[node_p['label']].update(node_p)

            # Layout positions (spring layout is generic; you could use shell, kamada_kawai, etc)
            pos = nx.spring_layout(G, seed=42)

            # Edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='black'),
                hoverinfo='none',
                mode='lines'
            )

            # Nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_info = G.nodes[node]
                node_text.append(str(node_info.get('label', node)))
                node_color.append(node_info.get('fillcolor', '#BBBBBB'))

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=40, color=node_color, line=dict(width=2, color='black')),
                text=node_text,
                textposition="middle center",
                hoverinfo='text'
            )

            # Edge labels
            edge_label_x = []
            edge_label_y = []
            edge_label_text = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_label_x.append((x0 + x1) / 2)
                edge_label_y.append((y0 + y1) / 2)
                edge_label_text.append(edge_labels.get((edge[0], edge[1]), ""))

            edge_label_trace = go.Scatter(
                x=edge_label_x, y=edge_label_y,
                mode='text',
                text=edge_label_text,
                textposition='top center',
                hoverinfo='none',
                showlegend=False
            )

            fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0,l=0,r=0,t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor='white'
                            ))

            # Embed Plotly as HTML div
            from plotly.io import to_html
            fig_id = str(uuid.uuid4())[:8]
            fig_html = to_html(fig, include_plotlyjs='cdn', full_html=False, div_id=fig_id)
            # Optionally wrap in a <div id=...> if you want to target this chart in the DOM

            # Return the figure in the response
            message_dict.update({'message': 'Flow chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_html}), 'message':message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''}), 'message':message}}

        return response

    def display_image(self, result):
        if result['status'] == 'success':
            # Parse the JSON response
            response_data = json.loads(result['response']['data'])
            plot_html = response_data['figure']

            # 3. Render the HTML in the Colab output
            display(HTML(plot_html))
        else:
            # Handle the error message if the plot failed
            print(f"Failed to plot: {result['response']['message']}")           