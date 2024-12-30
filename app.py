import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from autoviz.AutoViz_Class import AutoViz_Class
from fpdf import FPDF
import openai

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/feature_visualizations'
AUTOVIZ_FOLDER = 'static/autoviz_plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUTOVIZ_FOLDER, exist_ok=True)

# Load the Iris Dataset
data = load_iris(as_frame=True)
df = data['frame']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Data Preprocessing: Cleaning the dataset
    if df.isnull().sum().sum() > 0:
        print("Missing values detected. Filling with mean values.")
        df.fillna(df.mean(), inplace=True)

    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    if df.shape != initial_shape:
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Save the cleaned dataset to a CSV file
    cleaned_data_path = "static/cleaned_dataset.csv"
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_data_path}")

    # Split the data
    X = df.drop(columns=['target'])
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    is_classification = len(y.unique()) <= 20

    # Lazy Predict Model Comparison
    if is_classification:
        lazy_model = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
    else:
        lazy_model = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)

    models, predictions = lazy_model.fit(X_train, X_test, y_train, y_test)
    best_model_name = models.index[0]

    # Feature Importance Analysis
    if is_classification:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_names = X.columns
    important_features = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    important_features = important_features.sort_values(by='Importance', ascending=False)

    # Generate Visualizations for Top Features
    top_features = important_features.head(3)['Feature']
    for feature in top_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='target', y=feature, palette='Set2')
        plt.title(f"Boxplot of {feature} by Target")
        plt.savefig(os.path.join(UPLOAD_FOLDER, f"boxplot_{feature}.png"))
        plt.close()

    # AutoViz Analysis
    AV = AutoViz_Class()
    AV.AutoViz(
        filename=None,
        sep=",",
        dfte=df,
        depVar="target",
        save_plot_dir=AUTOVIZ_FOLDER
    )

    # Generate Textual Summary
    data_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "descriptive_stats": df.describe().to_dict(),
    }
    textual_summary = generate_textual_summary(data_summary)

    # Generate PDF Report
    pdf_path = generate_pdf_report(textual_summary, UPLOAD_FOLDER, AUTOVIZ_FOLDER)
    
    return render_template('index.html', best_model=best_model_name, pdf_report=pdf_path, plots=os.listdir(UPLOAD_FOLDER), cleaned_data=cleaned_data_path)

def generate_textual_summary(data_summary):
    prompt = f"""
    Write a detailed project report in simple English based on the following dataset analysis:
    1. The dataset has {data_summary['shape'][0]} rows and {data_summary['shape'][1]} columns.
    2. The column names are: {', '.join(data_summary['columns'])}.
    3. Missing values: {data_summary['missing_values']}.
    4. Data types: {data_summary['data_types']}.
    5. Descriptive statistics: {data_summary['descriptive_stats']}.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_pdf_report(summary, viz_folder, autoviz_folder):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)

    for plot_file in os.listdir(viz_folder):
        pdf.add_page()
        pdf.image(os.path.join(viz_folder, plot_file), x=10, y=None, w=180)

    for plot_file in os.listdir(autoviz_folder):
        if plot_file.endswith(".svg"):
            pdf.add_page()
            pdf.image(os.path.join(autoviz_folder, plot_file), x=10, y=None, w=180)

    pdf_path = "static/Complete_Analysis_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

if __name__ == '__main__':
    app.run(debug=True)
