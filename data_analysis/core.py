from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
from scipy.stats import chi2_contingency


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
        
        numerical_df = self.df.select_dtypes(exclude=[np.number])
        corr = numerical_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                        title="Pearson Correlation Heatmap")
        fig.show()

    def get_categorical_correlation(self):
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
                color_continuous_scale="viridis",
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