from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files



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

    def extract_normalized_categorical_data(self):
        """
        Extracts categorical columns and maps values to a uniform range [0, 1].
        Example: 3 categories become [0.0, 0.5, 1.0].
        """
        if self.df is None: return print("Error: No data loaded.")

        # Select only categorical columns
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()

        if cat_df.empty:
            return print("⚠️ No categorical columns found to transform.")

        for col in cat_df.columns:
            # Convert to category codes (0, 1, 2...)
            codes = cat_df[col].astype('category').cat.codes
            max_code = codes.max()

            # Normalize to [0, 1] range
            if max_code > 0:
                cat_df[col] = codes / max_code
            else:
                cat_df[col] = 0.0  # Handle columns with only one unique value
        self.categorical_normalized_df = cat_df
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

    def plot_correlation_heatmap(self):
        """
        Displays an interactive Heatmap of the Pearson Correlation matrix
        for all numeric features in the dataset.
        """
        if self.df is None: return
        
        if self.normalized_data_df is None:
            inspector.create_normalized_data_df() 
        corr = self.normalized_data_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                        title="Pearson Correlation Heatmap")
        fig.show()


# Initialize the interactive inspector
inspector = DataInspector()