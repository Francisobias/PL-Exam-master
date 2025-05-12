import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
import os
from io import BytesIO
import base64
from docx import Document
import numpy as np
from scipy.stats import chi2_contingency

app = Flask(__name__)

# Clean data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(how='all')  # Drop rows with all missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')  # Fill missing categorical with 'Unknown'
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())  # Fill missing numeric with median
    return df

# Read .docx file
def read_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

# Load data for a dataset
def load_data(dataset_num):
    try:
        if dataset_num < 5:
            csv_path = f'dataset_{dataset_num}/matches_{dataset_num}.csv'
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            csv_files = [f'dataset_5/matches_5_{j}.csv' for j in range(1, 11)]
            combined_df = pd.DataFrame()
            files_read = 0
            for file in csv_files:
                if not os.path.exists(file):
                    print(f"Warning: File not found: {file}")
                    continue
                try:
                    df = pd.read_csv(file)
                    df.columns = df.columns.str.lower().str.replace(' ', '_')
                    # Align columns
                    if not combined_df.empty:
                        for col in combined_df.columns:
                            if col not in df.columns:
                                df[col] = pd.NA
                        for col in df.columns:
                            if col not in combined_df.columns:
                                combined_df[col] = pd.NA
                    combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)
                    files_read += 1
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
            if files_read == 0:
                raise ValueError("No valid CSV files found for Dataset 5")
            if combined_df.empty:
                raise ValueError("Combined Dataset 5 is empty after processing")
            df = combined_df
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return clean_data(df)
    except Exception as e:
        print(f"Error loading dataset {dataset_num}: {str(e)}")
        return None

# Select the most relevant numeric column based on variance
def select_numeric_column(df, num_cols):
    if not num_cols:
        return None
    variances = df[num_cols].var()
    return variances.idxmax() if not variances.empty and variances.max() > 0 else num_cols[0]

# Select the most relevant categorical column based on value counts
def select_categorical_column(df, cat_cols):
    if not cat_cols:
        return None
    value_counts = [df[col].value_counts().iloc[:10].sum() for col in cat_cols]
    return cat_cols[np.argmax(value_counts)] if value_counts else cat_cols[0]

# Select the most correlated numeric pair
def select_numeric_pair(df, num_cols):
    if len(num_cols) < 2:
        return None, None
    corr = df[num_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_corr = corr.max().max()
    if max_corr > 0:
        pair = corr.stack().idxmax()
        return pair[0], pair[1], max_corr
    return num_cols[0], num_cols[1], 0

# Select the most associated categorical pair (chi-squared test)
def select_categorical_pair(df, cat_cols):
    if len(cat_cols) < 2:
        return None, None, 1
    best_pair = None
    min_p = 1
    for i, col1 in enumerate(cat_cols):
        for col2 in cat_cols[i+1:]:
            ctab = pd.crosstab(df[col1], df[col2])
            if ctab.size > 0 and ctab.shape[0] > 1 and ctab.shape[1] > 1:
                try:
                    _, p, _, _ = chi2_contingency(ctab)
                    if p < min_p:
                        min_p = p
                        best_pair = (col1, col2)
                except:
                    continue
    return best_pair if best_pair else (cat_cols[0], cat_cols[1]), min_p

# Evaluate visualization options and score them
def evaluate_visualizations(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    scores = {}

    # Histogram: Suitable for numeric columns with high variance
    if num_cols:
        num_col = select_numeric_column(df, num_cols)
        variance = df[num_col].var() if num_col else 0
        scores['histogram'] = variance / (df[num_col].std() + 1) if variance > 0 else 0
    else:
        scores['histogram'] = 0

    # Count plot: Suitable for categorical columns with high frequency
    if cat_cols:
        cat_col = select_categorical_column(df, cat_cols)
        count_score = df[cat_col].value_counts().iloc[:10].sum() / len(df) if cat_col else 0
        scores['count_plot'] = count_score if df[cat_col].nunique() <= 20 else count_score * 0.5
    else:
        scores['count_plot'] = 0

    # Scatter plot: Suitable for correlated numeric pairs
    if len(num_cols) >= 2:
        x_col, y_col, corr = select_numeric_pair(df, num_cols)
        scores['scatter_plot'] = corr if corr > 0.1 else corr * 0.5
    else:
        scores['scatter_plot'] = 0

    # Heatmap: Suitable for multiple numeric columns with correlations
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        scores['heatmap'] = corr_matrix.mean().mean() if not corr_matrix.empty else 0
    else:
        scores['heatmap'] = 0

    # Stacked bar: Suitable for categorical pairs with strong association
    if len(cat_cols) >= 2:
        (cat_col1, cat_col2), p = select_categorical_pair(df, cat_cols)
        assoc_score = 1 - p if p < 1 else 0
        nunique1 = df[cat_col1].nunique() if cat_col1 else float('inf')
        nunique2 = df[cat_col2].nunique() if cat_col2 else float('inf')
        scores['stacked_bar'] = assoc_score if nunique1 <= 20 and nunique2 <= 20 else assoc_score * 0.5
    else:
        scores['stacked_bar'] = 0

    return scores

# Create visualization based on selected type
def create_visualization(df, label, viz_type):
    if not os.path.exists('static'):
        os.makedirs('static')

    img = BytesIO()
    try:
        plt.figure(figsize=(12, 8))
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if viz_type == 'histogram':
            num_col = select_numeric_column(df, num_cols)
            if num_col:
                sns.histplot(df[num_col], kde=True, bins=30)
                plt.title(f'{label} - Distribution of {num_col}', fontsize=14, pad=15)
                plt.xlabel(num_col, fontsize=12)
                plt.ylabel('Count', fontsize=12)
            else:
                plt.text(0.5, 0.5, "No numeric columns for histogram", ha='center', va='center', fontsize=12)

        elif viz_type == 'count_plot':
            cat_col = select_categorical_column(df, cat_cols)
            if cat_col:
                top_categories = df[cat_col].value_counts().index[:10]
                sns.countplot(data=df[df[cat_col].isin(top_categories)], x=cat_col, order=top_categories)
                plt.title(f'{label} - Frequency of {cat_col}', fontsize=14, pad=15)
                plt.xlabel(cat_col, fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
            else:
                plt.text(0.5, 0.5, "No categorical columns for count plot", ha='center', va='center', fontsize=12)

        elif viz_type == 'scatter_plot':
            x_col, y_col, _ = select_numeric_pair(df, num_cols)
            if x_col and y_col:
                sns.scatterplot(x=x_col, y=y_col, data=df.sample(n=min(1000, len(df)), random_state=1), alpha=0.6)
                plt.title(f'{label} - Relationship between {x_col} and {y_col}', fontsize=14, pad=15)
                plt.xlabel(x_col, fontsize=12)
                plt.ylabel(y_col, fontsize=12)
            else:
                plt.text(0.5, 0.5, "Need at least two numeric columns for scatter plot", ha='center', va='center', fontsize=12)

        elif viz_type == 'heatmap':
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, annot_kws={'size': 10})
                plt.title(f'{label} - Correlation Heatmap of Numeric Variables', fontsize=14, pad=15)
            else:
                plt.text(0.5, 0.5, "Need at least two numeric columns for heatmap", ha='center', va='center', fontsize=12)

        elif viz_type == 'stacked_bar':
            cat_col1, cat_col2 = select_categorical_pair(df, cat_cols)[0]
            if cat_col1 and cat_col2 and df[cat_col1].nunique() <= 10 and df[cat_col2].nunique() <= 10:
                top_cat1 = df[cat_col1].value_counts().index[:10]
                top_cat2 = df[cat_col2].value_counts().index[:10]
                filtered_df = df[df[cat_col1].isin(top_cat1) & df[cat_col2].isin(top_cat2)]
                if not filtered_df.empty:
                    ctab = pd.crosstab(filtered_df[cat_col1], filtered_df[cat_col2])
                    ctab.plot(kind='bar', stacked=True)
                    plt.title(f'{label} - Relationship between {cat_col1} and {cat_col2}', fontsize=14, pad=15)
                    plt.xlabel(cat_col1, fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.legend(title=cat_col2, fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "No data available after filtering categories", ha='center', va='center', fontsize=12)
            else:
                plt.text(0.5, 0.5, f"Need two categorical columns with ≤10 unique values\nFound: {len(cat_cols)} columns, "
                                   f"{df[cat_col1].nunique() if cat_col1 else 0} and "
                                   f"{df[cat_col2].nunique() if cat_col2 else 0} unique values",
                         ha='center', va='center', fontsize=12)

        plt.tight_layout(pad=2.0)
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
        plt.close()

    except Exception as e:
        print(f"Error plotting {label}: {e}")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", ha='center', va='center', fontsize=12)
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
        plt.close()

    return plot

# Assign unique visualizations to each dataset
def assign_visualizations(datasets_scores):
    available_viz = ['histogram', 'count_plot', 'scatter_plot', 'heatmap', 'stacked_bar']
    assignments = {}
    used_viz = set()

    for dataset_num in range(1, 6):
        label = f'Dataset {dataset_num}'
        scores = datasets_scores.get(label, {})
        if not scores:
            assignments[label] = None
            continue
        # Sort visualizations by score, excluding used ones
        viz_options = [(viz, score) for viz, score in scores.items() if viz in available_viz and viz not in used_viz]
        if not viz_options:
            # Fallback to any unused visualization
            viz_options = [(viz, 0) for viz in available_viz if viz not in used_viz]
        if viz_options:
            best_viz = max(viz_options, key=lambda x: x[1])[0]
            assignments[label] = best_viz
            used_viz.add(best_viz)
        else:
            assignments[label] = None
    return assignments

@app.route('/')
def index():
    dataset_info = {}
    all_plots = {}
    error = None
    datasets_scores = {}

    try:
        for i in range(1, 6):
            label = f'Dataset {i}'
            docx_path = f'dataset_{i}/matches_{i}_example.docx'

            # Load data
            df = load_data(i)
            if df is None or df.empty:
                dataset_info[label] = {'context': f"No data available for Dataset {i}"}
                all_plots[label] = None
                continue

            # Read context from docx
            context = read_docx(docx_path)
            dataset_info[label] = {'context': context}

            # Evaluate visualizations
            datasets_scores[label] = evaluate_visualizations(df)

        # Assign unique visualizations
        viz_assignments = assign_visualizations(datasets_scores)

        # Generate plots
        for i in range(1, 6):
            label = f'Dataset {i}'
            if label not in dataset_info:
                continue
            df = load_data(i)
            if df is None or df.empty:
                continue
            viz_type = viz_assignments.get(label)
            if viz_type:
                all_plots[label] = create_visualization(df, label, viz_type)
            else:
                all_plots[label] = None
                dataset_info[label]['context'] += "\nNo suitable visualization available."

        return render_template('index.html', info=dataset_info, plots=all_plots, error=error)

    except Exception as e:
        error = f"Error processing datasets: {str(e)}"
        return render_template('index.html', info={}, plots={}, error=error)

if __name__ == '__main__':
    app.run(debug=True)