import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# App configuration
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample data loading functions
@st.cache_data
def load_sample_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

@st.cache_data
def generate_drift_data():
    dates = pd.date_range(end=datetime.today(), periods=100).to_list()
    values = np.random.normal(0, 1, 100).cumsum() + 10
    return pd.DataFrame({'date': dates, 'value': values})

# Navigation
def navigation():
    st.sidebar.title("Data Science Dashboard")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["ETL & Data Visualization", "Model Training", "Data & Model Drift", "Explainable AI (SHAP)"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Modernistic dashboard for end-to-end data science workflow. "
        "Includes ETL, modeling, monitoring, and explainability."
    )
    return page

def load_uploaded_file(uploaded_file):
    """Load either CSV or Excel file based on file extension"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                return pd.read_excel(uploaded_file, engine='openpyxl')
            except ImportError:
                st.error("""
                **Missing dependency**: The 'openpyxl' package is required to read Excel files.
                
                Please install it by running:
                ```
                pip install openpyxl
                ```
                Then restart the app.
                """)
                return None
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Page 1: ETL & Data Visualization
def etl_data_viz():
    st.title("📥 ETL & Data Visualization")
    st.markdown("Extract, Transform, Load processes with interactive visualizations")
    
    with st.expander("Upload Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=['csv', 'xls', 'xlsx'],
            help="Upload your dataset in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            df = load_uploaded_file(uploaded_file)
            if df is None:
                st.stop()  # Stop execution if file loading failed
            st.success("File successfully loaded!")
        else:
            df = load_sample_data()
            st.info("Using sample breast cancer dataset. Upload your own file to explore your data.")
    
    # Only proceed if df exists and is not None
    if df is None or df.empty:
        st.warning("No valid data loaded. Please upload a file or use sample data.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Column Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap"]
    )
    
    if viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", df.columns)
        with col2:
            y_axis = st.selectbox("Y-axis", df.columns)
        
        color_by = st.selectbox("Color by", [None] + list(df.select_dtypes(include=['object', 'category', 'int']).columns))
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                         hover_data=df.columns, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox("Select column", df.select_dtypes(include=['number']).columns)
        with col2:
            bins = st.slider("Number of bins", 5, 100, 20)
        
        fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox("Select column", df.select_dtypes(include=['number']).columns)
        with col2:
            group_by = st.selectbox("Group by", [None] + list(df.select_dtypes(include=['object', 'category']).columns))
        
        fig = px.box(df, y=column, x=group_by, title=f"Box Plot of {column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 0:
            fig = px.imshow(numeric_df.corr(), 
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale='Viridis',
                            title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for correlation heatmap")

# Page 2: Model Training
def model_training():
    st.title("🤖 Model Training")
    st.markdown("Train and evaluate machine learning models")
    
    df = load_sample_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    st.subheader("Dataset Information")
    st.markdown(f"""
    - **Samples**: {X.shape[0]}
    - **Features**: {X.shape[1]}
    - **Target distribution**: {y.value_counts().to_dict()}
    """)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        random_state = st.number_input("Random state", 0, 100, 42)
    
    with col2:
        n_estimators = st.slider("Number of estimators", 10, 200, 100)
        max_depth = st.slider("Max depth", 2, 20, 5)
    
    st.markdown("---")
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.success("Model trained successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_score:.2%}")
            with col2:
                st.metric("Test Accuracy", f"{test_score:.2%}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Feature Importances")
            ax.barh(range(10), importances[indices][:10], align="center")
            ax.set_yticks(range(10))
            ax.set_yticklabels(X.columns[indices][:10])
            ax.invert_yaxis()
            st.pyplot(fig)
    
    if 'model' not in st.session_state:
        st.warning("Configure your model and click 'Train Model' to get started")

# Page 3: Data & Model Drift
def data_model_drift():
    st.title("📉 Data & Model Drift")
    st.markdown("Monitor data and model performance over time")
    
    st.subheader("Data Drift Detection")
    drift_df = generate_drift_data()
    
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Moving average window", 5, 30, 10)
    with col2:
        threshold = st.slider("Drift threshold", 0.1, 2.0, 0.5)
    
    drift_df['ma'] = drift_df['value'].rolling(window=window_size).mean()
    drift_df['upper'] = drift_df['ma'] + threshold
    drift_df['lower'] = drift_df['ma'] - threshold
    
    fig = px.line(drift_df, x='date', y=['value', 'ma', 'upper', 'lower'],
                  title="Data Drift Monitoring", 
                  labels={'value': 'Metric Value', 'date': 'Date'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Model Performance Drift")
    
    # Simulate performance metrics over time
    dates = pd.date_range(end=datetime.today(), periods=30).to_list()
    accuracy = np.clip(np.random.normal(0.85, 0.05, 30).cumsum() / np.arange(1, 31), 0.7, 0.95)
    precision = np.clip(accuracy - np.random.normal(0.05, 0.01, 30), 0.6, 0.95)
    recall = np.clip(accuracy + np.random.normal(0.03, 0.01, 30), 0.65, 0.98)
    
    perf_df = pd.DataFrame({
        'date': dates,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })
    
    metric = st.selectbox("Select metric to monitor", ['accuracy', 'precision', 'recall'])
    
    fig = px.line(perf_df, x='date', y=metric,
                  title=f"Model {metric.capitalize()} Over Time",
                  markers=True)
    
    # Add reference line
    ref_line = perf_df[metric].mean()
    fig.add_hline(y=ref_line, line_dash="dash", line_color="red",
                  annotation_text=f"Average: {ref_line:.2f}",
                  annotation_position="bottom right")
    
    st.plotly_chart(fig, use_container_width=True)

# Page 4: Explainable AI (SHAP)
def explainable_ai():
    st.title("🔍 Explainable AI (SHAP)")
    st.markdown("Model interpretability using SHAP values")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model on the 'Model Training' page first")
        return
    
    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    
    st.subheader("Global Feature Importance")
    
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary/multi-class cases
        if isinstance(shap_values, list):
            # For binary classification, use index 1 (positive class)
            if len(shap_values) == 2:
                shap_values = shap_values[1]
                expected_value = explainer.expected_value[1]
            else:
                # For multi-class, use first class (can modify as needed)
                shap_values = shap_values[0]
                expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
        
        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        
        st.markdown("---")
        st.subheader("Individual Prediction Explanation")
        
        sample_idx = st.slider("Select sample to explain", 0, len(X_test)-1, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sample Features")
            st.dataframe(X_test.iloc[[sample_idx]], use_container_width=True)
            
            actual = y_test.iloc[sample_idx]
            pred = model.predict(X_test.iloc[[sample_idx]])[0]
            
            st.metric("Actual", actual)
            st.metric("Predicted", pred)
        
        with col2:
            st.markdown("#### SHAP Force Plot")
            plt.figure()
            
            # Use the new API format
            shap.plots.force(
                expected_value,
                shap_values[sample_idx, :],
                X_test.iloc[sample_idx, :],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf(), bbox_inches='tight', clear_figure=True)
            
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {str(e)}")
        st.info("This might occur if the model hasn't been properly trained or if there are feature mismatches.")

# Main app logic
def main():
    page = navigation()
    
    if page == "ETL & Data Visualization":
        etl_data_viz()
    elif page == "Model Training":
        model_training()
    elif page == "Data & Model Drift":
        data_model_drift()
    elif page == "Explainable AI (SHAP)":
        explainable_ai()

if __name__ == "__main__":
    main()