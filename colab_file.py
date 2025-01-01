

# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from autoviz.AutoViz_Class import AutoViz_Class
from fpdf import FPDF
import openai
from PIL import Image

# Define output directories for saving reports and visualizations
OUTPUT_FOLDER = "feature_visualizations"
AUTOVIZ_OUTPUT_FOLDER = "autoviz_reports"
REPORT_FILE = "Complete_Textual_Summary_Report.pdf"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AUTOVIZ_OUTPUT_FOLDER, exist_ok=True)

# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    data = load_iris(as_frame=True)
    df = data['frame']

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values detected. Filling with mean values.")
        df = df.fillna(df.mean())

    # Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    return df

# Train model and calculate feature importances
def train_model_and_get_importances(df):
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Using RandomForest for classification
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    feature_importances = model.feature_importances_
    feature_names = df.drop(columns=['target']).columns
    important_features = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    important_features = important_features.sort_values(by='Importance', ascending=False)
    
    return important_features

# Generate feature visualizations
def create_feature_plots(df, important_features):
    top_features = important_features.head(3)['Feature']
    print(f"Top Features for Visualization: {list(top_features)}")

    for feature in top_features:
        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='target', y=feature, palette='Set2')
        plt.title(f"Boxplot of {feature} by Target")
        plt.xlabel("Target")
        plt.ylabel(feature)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"boxplot_{feature}.png"), dpi=300)
        plt.close()

        # Violin Plot
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x='target', y=feature, palette='Set3')
        plt.title(f"Violin Plot of {feature} by Target")
        plt.xlabel("Target")
        plt.ylabel(feature)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"violinplot_{feature}.png"), dpi=300)
        plt.close()

        # Scatterplot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature, y='target', hue='target', palette='viridis', s=100)
        plt.title(f"Scatterplot of {feature} vs Target")
        plt.xlabel(feature)
        plt.ylabel("Target")
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"scatterplot_{feature}.png"), dpi=300)
        plt.close()

# Generate AutoViz report
def generate_autoviz_report(df):
    AV = AutoViz_Class()
    df.to_csv("iris_dataset_cleaned.csv", index=False)
    AV.AutoViz(
        filename="iris_dataset_cleaned.csv",
        sep=",",
        depVar="target",
        header=0,
        verbose=1,
        lowess=False,
        chart_format="svg",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir=AUTOVIZ_OUTPUT_FOLDER
    )

# Convert SVG to PNG
def convert_svg_to_png(svg_path, png_path):
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=png_path)
    except ImportError:
        print("Please install cairosvg for SVG to PNG conversion.")
        raise

# Generate detailed summary using OpenAI
def generate_detailed_summary(data_summary):
    prompt = f"""
    Write a detailed project report in simple English based on the following dataset analysis:
    1. The dataset has {data_summary['shape'][0]} rows and {data_summary['shape'][1]} columns.
    2. The column names are: {', '.join(data_summary['columns'])}.
    3. Missing values: {data_summary['missing_values']}.
    4. Data types: {data_summary['data_types']}.
    5. Descriptive statistics: {data_summary['descriptive_stats']}.
    """
    openai.api_key = ""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    return response["choices"][0]["message"]["content"].strip()




# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os

# Load the Iris dataset
data = load_iris(as_frame=True)
df = data['frame']

# Data Preprocessing: Cleaning the dataset
# 1. Check for missing values and fill or drop them
if df.isnull().sum().sum() > 0:
    print("Missing values detected. Filling with mean values.")
    df = df.fillna(df.mean())

# 2. Remove duplicate rows if any
initial_shape = df.shape
df = df.drop_duplicates()
if df.shape != initial_shape:
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

# 3. Correlation Analysis
print("\nCorrelation Analysis:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Remove features with low correlation with target (optional, based on threshold)
correlation_threshold = 0.1
low_corr_features = correlation_matrix['target'][abs(correlation_matrix['target']) < correlation_threshold].index
if len(low_corr_features) > 0:
    print(f"\nRemoving low-correlation features: {list(low_corr_features)}")
    df = df.drop(columns=low_corr_features)

# 4. Split dataset into features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# 5. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Automatic Model Selection using LazyPredict
print("\nAutomatically Selecting the Best Model:")
is_classification = len(y.unique()) <= 20  # Treat as classification if unique labels are ≤ 20

if is_classification:
    # For classification tasks
    lazy_model = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
    models, predictions = lazy_model.fit(X_train, X_test, y_train, y_test)
else:
    # For regression tasks
    lazy_model = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)
    models, predictions = lazy_model.fit(X_train, X_test, y_train, y_test)

# Display model performance
print("\nModel Performance Comparison:")
print(models)

# Select the best model based on R-squared or accuracy
best_model_name = models.index[0]
print(f"\nBest Model Selected: {best_model_name}")

# Retrain the best model (optional)
# LazyPredict does not return trained models; this step would require manual implementation
if is_classification:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
else:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the selected model
print("\nEvaluation Report:")
if is_classification:
    print(classification_report(y_test, y_pred))
else:
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-Squared Score: {r2}")

# Notify user
print("\nPipeline complete. The best model has been selected and evaluated.")
# Generate PDF report
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Complete Data Analysis Report", 0, 1, "C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_plot(self, plot_path, title):
        self.add_page()
        self.chapter_title(title)
        self.image(plot_path, x=10, y=None, w=180)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Train model and get feature importances
    important_features = train_model_and_get_importances(df)

    # Create feature visualizations
    create_feature_plots(df, important_features)

    # Generate AutoViz report
    generate_autoviz_report(df)

    # Generate dataset summary
    data_summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "descriptive_stats": df.describe().to_dict(),
    }

    # Generate textual summary
    textual_summary = generate_detailed_summary(data_summary)

    # Generate and save PDF report
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add textual summary to the report
    pdf.add_page()
    pdf.chapter_title("Textual Summary of the Dataset")
    pdf.chapter_body(textual_summary)

    # Add visualizations to the PDF
    for root, dirs, files in os.walk(AUTOVIZ_OUTPUT_FOLDER):
        for file in files:
            if file.endswith(".svg"):
                svg_path = os.path.join(root, file)
                png_path = svg_path.replace(".svg", ".png")
                convert_svg_to_png(svg_path, png_path)  # Convert to PNG
                pdf.add_plot(png_path, file.replace("_", " ").replace(".svg", "").title())

    # Save the final report
    pdf.output(REPORT_FILE)
    print(f"Report generated: {REPORT_FILE}")






