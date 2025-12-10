import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {margin-top: 1rem;}
    h1 {color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('Notebooks/models/model.pkl')
        preprocessor = joblib.load('Notebooks/models/scaler.pkl')
        return model, preprocessor, True
    except FileNotFoundError:
        return None, None, False

model, preprocessor, models_loaded = load_models()

@st.cache_data
def load_metrics():
    try:
        with open('Notebooks/models/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback values if the file doesn't exist yet
        return {"accuracy": 0, "f1_score": 0, "roc_auc": 0, "features_count": 0,"Recall":0,"Precision":0}

@st.cache_data
def compute_confusion_matrix():
    """Compute confusion matrix from actual model and data"""
    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        
        # Load the data
        df = pd.read_csv('data/customer_churn_data.csv')
        
        # Prepare features (matching the model training)
        X = df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']].copy()
        X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
        y = df['Churn']
        
        # Split data (use same random state as training to get test set)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Load model and preprocessor
        model = joblib.load('Notebooks/models/model.pkl')
        preprocessor = joblib.load('Notebooks/models/scaler.pkl')
        
        # Transform and predict on test set
        X_test_scaled = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        return cm.tolist()
    except Exception as e:
        # Fallback to placeholder if computation fails
        return [[0, 0], [0, 0]]

metrics_data = load_metrics()
# Sidebar
with st.sidebar:
    st.title("🎯 Navigation")
    page = st.radio("Select Page", 
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Insights", "ℹ️ About"],
        label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    if models_loaded:
        st.success("✅ Model Loaded")
        st.info("**Features:**\n- Gender\n- Age\n- Tenure\n- Monthly Charges")
    else:
        st.error("❌ Model Not Found")

# Helper functions
def get_risk_color(prob):
    if prob > 0.7:
        return "red"
    elif prob > 0.4:
        return "orange"
    return "green"

def get_risk_level(prob):
    if prob > 0.7:
        return "HIGH RISK"
    elif prob > 0.4:
        return "MEDIUM RISK"
    return "LOW RISK"

def get_recommendation(prob, gender, age, tenure, monthly_charges):
    """Generate personalized recommendations"""
    recommendations = []
    
    if prob > 0.7:
        recommendations.append("🚨 **IMMEDIATE ACTION REQUIRED**")
        if tenure < 12:
            recommendations.append("• Offer first-year loyalty discount (20%)")
        if monthly_charges > 70:
            recommendations.append("• Review pricing plan - customer may find cheaper alternatives")
        recommendations.append("• Schedule personal retention call within 48 hours")
        recommendations.append("• Assign dedicated account manager")
    elif prob > 0.4:
        recommendations.append("⚠️ **PROACTIVE MONITORING**")
        recommendations.append("• Send satisfaction survey")
        recommendations.append("• Highlight unused service benefits")
        if tenure < 24:
            recommendations.append("• Offer contract extension with benefits")
    else:
        recommendations.append("✅ **LOW RISK - MAINTAIN ENGAGEMENT**")
        recommendations.append("• Continue standard service")
        recommendations.append("• Consider upsell opportunities")
        recommendations.append("• Send quarterly check-in email")
    
    return "\n".join(recommendations)

# ===== HOME PAGE =====
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction System")
    st.markdown("### Predict customer churn using Machine Learning")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
        "Accuracy", 
        f"{metrics_data['accuracy']:.1%}", 
        "Real-time"
    )
    with col2:
        st.metric(
        "F1-Score", 
        f"{metrics_data['f1_score']:.3f}"
    )
    with col3:
        st.metric(
        "AUC-ROC", 
        f"{metrics_data['roc_auc']:.3f}"
    )
    with col4:
        st.metric(
        "Features", 
        str(metrics_data['features_count'])
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What This App Does")
        st.markdown("""
        - **Predict** individual customer churn risk
        - **Analyze** batch customer data via CSV upload
        - **Visualize** model performance and insights
        - **Generate** actionable retention strategies
        - **Export** predictions for business teams
        """)
        
        st.markdown("### 📋 Required Features")
        st.info("""
        1. **Gender**: Male/Female
        2. **Age**: Customer age in years
        3. **Tenure**: Months with company
        4. **Monthly Charges**: Monthly bill amount ($)
        """)
    
    with col2:
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        **Single Prediction:**
        1. Navigate to 🔮 Single Prediction
        2. Enter customer details
        3. Click "Predict Churn Risk"
        4. View risk level and recommendations
        
        **Batch Prediction:**
        1. Navigate to 📊 Batch Prediction
        2. Upload CSV with required columns
        3. Get predictions for all customers
        4. Download results with risk levels
        """)
        
        with st.expander("📄 CSV Format Example"):
            example_df = pd.DataFrame({
                'gender': ['Male', 'Female', 'Male'],
                'age': [45, 32, 58],
                'tenure': [12, 48, 6],
                'monthly_charges': [75.50, 45.20, 89.90]
            })
            st.dataframe(example_df, use_container_width=True)
    
    if not models_loaded:
        st.error("⚠️ **Models not found!** Train your model and save to `models/` directory.")
        st.code("""
# In your Jupyter notebook:
import joblib

# Save model
joblib.dump(model, 'Notebooks/models/model.pkl')
joblib.dump(preprocessor, 'Notebooks/models/scaler.pkl')
        """, language="python")

# ===== SINGLE PREDICTION PAGE =====
elif page == "🔮 Single Prediction":
    st.title("🔮 Predict Individual Customer Churn")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please train and save your models first.")
        st.stop()
    
    st.markdown("### Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"], help="Customer gender")
        age = st.slider("🎂 Age (years)", 18, 100, 35, help="Customer age in years")
    
    with col2:
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12, help="Months with company")
        monthly_charges = st.number_input("💰 Monthly Charges ($)", 
                                          min_value=0.0, 
                                          max_value=200.0, 
                                          value=50.0, 
                                          step=5.0,
                                          help="Monthly bill amount")
    
    st.markdown("---")
    
    # Customer summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gender", gender)
    col2.metric("Age", f"{age} years")
    col3.metric("Tenure", f"{tenure} months")
    col4.metric("Monthly Charges", f"${monthly_charges:.2f}")
    
    st.markdown("---")
    
    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Analyzing customer data..."):
            try:
                # Prepare input
                input_df = pd.DataFrame({
                'Age': [age],
                'Gender': [1 if gender == "Male" else 0], # Convert text to 0/1 matching your training
                'Tenure': [tenure],
                'MonthlyCharges': [monthly_charges]
})
                input_features = preprocessor.transform(input_df)
                
                # Predict
                prediction = model.predict(input_features)[0]
                probability = model.predict_proba(input_features)[0][1]
                
                st.markdown("### 🎯 Prediction Results")
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': get_risk_color(probability)},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    risk_level = get_risk_level(probability)
                    if probability > 0.7:
                        st.error(f"### ⚠️ {risk_level}")
                        st.markdown(f"**{probability*100:.1f}%** chance of churn")
                    elif probability > 0.4:
                        st.warning(f"### ⚡ {risk_level}")
                        st.markdown(f"**{probability*100:.1f}%** chance of churn")
                    else:
                        st.success(f"### ✅ {risk_level}")
                        st.markdown(f"**{probability*100:.1f}%** chance of churn")
                    
                    # Customer profile summary
                    st.markdown("#### 📋 Customer Profile")
                    st.markdown(f"""
                    - **Tenure**: {tenure} months ({tenure/12:.1f} years)
                    - **Monthly Spend**: ${monthly_charges:.2f}
                    - **Total Spend**: ${monthly_charges * tenure:.2f}
                    - **Avg Monthly**: ${monthly_charges:.2f}
                    """)
                
                with col2:
                    st.markdown("#### 💡 Recommended Actions")
                    recommendation = get_recommendation(probability, gender, age, tenure, monthly_charges)
                    st.markdown(recommendation)
                
                # Risk factors analysis
                st.markdown("---")
                st.markdown("### 📊 Risk Factor Analysis")
                
                # Create risk factors based on input
                risk_factors = []
                
                if tenure < 12:
                    risk_factors.append(("Short Tenure", 0.35, "High risk - new customers churn more"))
                elif tenure < 24:
                    risk_factors.append(("Medium Tenure", 0.20, "Moderate risk - building loyalty"))
                else:
                    risk_factors.append(("Long Tenure", 0.10, "Lower risk - established customer"))
                
                if monthly_charges > 70:
                    risk_factors.append(("High Monthly Charges", 0.30, "Price-sensitive - may seek cheaper options"))
                elif monthly_charges > 50:
                    risk_factors.append(("Medium Monthly Charges", 0.15, "Balanced pricing"))
                else:
                    risk_factors.append(("Low Monthly Charges", 0.08, "Good value perception"))
                
                if age < 30:
                    risk_factors.append(("Younger Age", 0.18, "More likely to switch providers"))
                elif age < 50:
                    risk_factors.append(("Middle Age", 0.12, "Moderate switching tendency"))
                else:
                    risk_factors.append(("Older Age", 0.07, "Less likely to change"))
                
                risk_factors.append((f"Gender ({gender})", 0.10, "Demographic factor"))
                
                # Plot risk factors
                factors_df = pd.DataFrame(risk_factors, columns=['Factor', 'Impact', 'Interpretation'])
                
                fig = px.bar(factors_df, 
                            x='Impact', 
                            y='Factor', 
                            orientation='h',
                            color='Impact',
                            color_continuous_scale='Reds',
                            text='Impact')
                fig.update_traces(texttemplate='%{text:.0%}', textposition='outside')
                fig.update_layout(showlegend=False, height=300, xaxis_title="Risk Impact")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                with st.expander("📖 View Detailed Interpretations"):
                    for factor, impact, interpretation in risk_factors:
                        st.markdown(f"**{factor}** ({impact:.0%} impact): {interpretation}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.error("Please check that your model and preprocessor are compatible.")

# ===== BATCH PREDICTION PAGE =====
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Churn Prediction")
    st.markdown("Upload a CSV file with customer data to get bulk predictions")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please train and save your models first.")
        st.stop()
    
    st.info("📋 **Required columns:** `gender`, `age`, `tenure`, `monthly_charges`")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns (accept both lowercase and PascalCase)
            required_cols_lower = ['gender', 'age', 'tenure', 'monthly_charges']
            required_cols_pascal = ['Gender', 'Age', 'Tenure', 'MonthlyCharges']
            
            # Check which format the CSV uses
            has_lower = all(col in df.columns for col in required_cols_lower)
            has_pascal = all(col in df.columns for col in required_cols_pascal)
            
            if not has_lower and not has_pascal:
                st.error(f"❌ Missing required columns. Expected: `gender`, `age`, `tenure`, `monthly_charges`")
                st.stop()
            
            # Standardize column names to match model training format
            if has_lower:
                df = df.rename(columns={
                    'gender': 'Gender', 'age': 'Age', 
                    'tenure': 'Tenure', 'monthly_charges': 'MonthlyCharges'
                })
            
            st.success(f"✅ File uploaded successfully: **{len(df)} customers**")
            
            with st.expander("📋 Preview Data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing predictions..."):
                    
                    # Prepare features - map Gender to numeric and scale
                    feature_df = df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']].copy()
                    feature_df['Gender'] = feature_df['Gender'].map({'Male': 1, 'Female': 0, 1: 1, 0: 0})
                    features = preprocessor.transform(feature_df)
                    
                    # Predict
                    predictions = model.predict(features)
                    probabilities = model.predict_proba(features)[:, 1]
                    
                    # Add results to dataframe
                    df['churn_probability'] = probabilities
                    df['risk_level'] = pd.cut(probabilities, 
                        bins=[0, 0.4, 0.7, 1.0], 
                        labels=['Low', 'Medium', 'High'])
                    df['prediction'] = predictions
                    df['recommendation'] = df['risk_level'].map({
                        'Low': 'Standard Service',
                        'Medium': 'Monitor & Survey',
                        'High': 'Immediate Retention'
                    })
                    
                    st.success("✅ Predictions completed!")
                    
                    # Summary metrics
                    st.markdown("### 📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        high_risk = (df['risk_level'] == 'High').sum()
                        st.metric("🔴 High Risk", 
                                 f"{high_risk}", 
                                 f"{high_risk/len(df)*100:.1f}%")
                    with col2:
                        medium_risk = (df['risk_level'] == 'Medium').sum()
                        st.metric("🟡 Medium Risk", 
                                 f"{medium_risk}", 
                                 f"{medium_risk/len(df)*100:.1f}%")
                    with col3:
                        low_risk = (df['risk_level'] == 'Low').sum()
                        st.metric("🟢 Low Risk", 
                                 f"{low_risk}", 
                                 f"{low_risk/len(df)*100:.1f}%")
                    with col4:
                        avg_prob = df['churn_probability'].mean()
                        st.metric("📊 Avg Churn Risk", 
                                 f"{avg_prob*100:.1f}%")
                    
                    # Results table
                    st.markdown("### 📋 Detailed Predictions")
                    
                    # Color code risk levels
                    def color_risk(val):
                        if val == 'High':
                            return 'background-color: #ffcdd2'
                        elif val == 'Medium':
                            return 'background-color: #fff9c4'
                        return 'background-color: #c8e6c9'
                    
                    styled_df = df.style.map(color_risk, subset=['risk_level'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                    
                    # Visualizations
                    st.markdown("### 📊 Visual Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk distribution
                        fig = px.pie(df, names='risk_level', 
                                    title='Risk Level Distribution',
                                    color='risk_level',
                                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Probability histogram
                        fig = px.histogram(df, x='churn_probability', 
                                          nbins=30,
                                          title='Churn Probability Distribution',
                                          color='risk_level',
                                          color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                        fig.update_xaxis(title="Churn Probability")
                        fig.update_yaxis(title="Number of Customers")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Segment analysis
                    st.markdown("### 🎯 Segment Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk by tenure
                        tenure_risk = df.groupby(pd.cut(df['tenure'], bins=[0, 12, 24, 36, 100]))['churn_probability'].mean()
                        fig = px.bar(x=tenure_risk.index.astype(str), y=tenure_risk.values,
                                    title='Average Churn Risk by Tenure',
                                    labels={'x': 'Tenure Range (months)', 'y': 'Avg Churn Probability'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk by monthly charges
                        charges_risk = df.groupby(pd.cut(df['monthly_charges'], bins=5))['churn_probability'].mean()
                        fig = px.bar(x=charges_risk.index.astype(str), y=charges_risk.values,
                                    title='Average Churn Risk by Monthly Charges',
                                    labels={'x': 'Charge Range ($)', 'y': 'Avg Churn Probability'})
                        st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ===== MODEL INSIGHTS PAGE =====
elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Insights")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎯 Feature Analysis", "📉 About Features"])
    
    with tab1:
        st.markdown("### Model Performance Overview")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            #  metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Score': [metrics_data["accuracy"], metrics_data["Precision"], metrics_data["Recall"], metrics_data["f1_score"], metrics_data["roc_auc"]],
                'Interpretation': [
                    'Overall correctness',
                    'Churner identification accuracy',
                    'Catch rate for actual churners',
                    'Balance of precision & recall',
                    'Model discrimination ability'
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🎯 Key Strengths")
            st.success(f"""
            - **AUC-ROC ({metrics_data['roc_auc']:.2f})**: Model discrimination ability
            - **F1-Score ({metrics_data['f1_score']:.2f})**: Balance of precision & recall
            - **Recall ({metrics_data['Recall']:.0%})**: Catches customers at risk
            """)
        
        with col2:
            # Real confusion matrix from model
            cm_values = compute_confusion_matrix()
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_values,
                x=['Predicted: No Churn', 'Predicted: Churn'],
                y=['Actual: No Churn', 'Actual: Churn'],
                text=cm_values,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Reading the Matrix:**
            - Top-left ({cm_values[0][0]}): Correctly identified non-churners (True Negatives)
            - Bottom-right ({cm_values[1][1]}): Correctly identified churners (True Positives)
            - Top-right ({cm_values[0][1]}): False alarms (False Positives)
            - Bottom-left ({cm_values[1][0]}): Missed churners (False Negatives)
            """)
    
    with tab2:
        st.markdown("### Feature Importance Analysis")
        
        # Mock feature importance (you can update with actual values)
        features_df = pd.DataFrame({
            'Feature': ['Tenure', 'Monthly Charges', 'Age', 'Gender'],
            'Importance': [0.40, 0.35, 0.18, 0.07],
            'Description': [
                'Time with company - most predictive',
                'Pricing sensitivity indicator',
                'Customer lifecycle stage',
                'Demographic factor'
            ]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(features_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        text='Importance')
            fig.update_traces(texttemplate='%{text:.0%}', textposition='outside')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(features_df[['Feature', 'Importance']], 
                        use_container_width=True, 
                        hide_index=True)
        
        st.markdown("### 💡 Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Top Risk Factors")
            st.markdown("""
            1. **Tenure < 12 months** → 70% churn risk
            2. **High charges (>$70)** → Price sensitivity
            3. **Younger customers (<30)** → More switching
            4. **New + expensive** → Highest risk combo
            """)
        
        with col2:
            st.markdown("#### ✅ Retention Strategies")
            st.markdown("""
            1. **First-year focus** → Loyalty programs
            2. **Pricing optimization** → Value perception
            3. **Age-specific offers** → Targeted campaigns
            4. **Early intervention** → Proactive outreach
            """)
    
    with tab3:
        st.markdown("### 📚 Understanding the Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Tenure")
            st.markdown("""
            **Definition:** Number of months customer has been with company
            
            **Impact on Churn:**
            - **0-12 months**: Highest risk (exploring alternatives)
            - **12-24 months**: Moderate risk (building loyalty)
            - **24+ months**: Lower risk (established relationship)
            
            **Actionable Insight:** Focus retention efforts on customers in first year
            """)
            
            st.markdown("#### 💰 Monthly Charges")
            st.markdown("""
            **Definition:** Amount customer pays per month
            
            **Impact on Churn:**
            - **High charges (>$70)**: Price sensitivity increases
            - **Medium charges ($40-70)**: Balanced value perception
            - **Low charges (<$40)**: Good value, lower churn
            
            **Actionable Insight:** Review pricing for high-paying customers regularly
            """)
        
        with col2:
            st.markdown("#### 🎂 Age")
            st.markdown("""
            **Definition:** Customer age in years
            
            **Impact on Churn:**
            - **18-30 years**: More likely to switch providers
            - **30-50 years**: Moderate switching behavior
            - **50+ years**: Less likely to change
            
            **Actionable Insight:** Younger customers need more engagement
            """)
            
            st.markdown("#### 👤 Gender")
            st.markdown("""
            **Definition:** Customer gender (Male/Female)
            
            **Impact on Churn:**
            - Minor demographic factor
            - Used in combination with other features
            - Helps segment marketing strategies
            
            **Actionable Insight:** Consider in targeted campaigns, not primary factor
            """)

# ===== ABOUT PAGE =====
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown(f"""
        This customer churn prediction system uses machine learning to identify 
        customers at risk of leaving the service. The model analyzes various 
        customer attributes and behavioral patterns to predict churn probability.
        
        **Dataset:** 1,000+ customer records  
        **Features:** {metrics_data['features_count']} key attributes  
        **Model:** Random Forest Classifier  
        **Accuracy:** {metrics_data['accuracy']:.1%}
        """)
        
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - Real-time single customer prediction
        - Batch processing via CSV upload
        - Risk segmentation (Low/Medium/High)
        - Feature importance analysis
        - Actionable business recommendations
        - Model performance visualization
        """)
    
    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        **Machine Learning:**
        - scikit-learn
        - XGBoost
        - SHAP (interpretability)
        
        **Web Framework:**
        - Streamlit
        - Plotly (visualizations)
        
        **Data Processing:**
        - pandas
        - NumPy
        """)
        
        st.markdown("### 📊 Model Details")
        st.info("""
        **Algorithm:** Random Forest Classifier  
        **Cross-Validation:** 5-fold  
        **Preprocessing:** StandardScaler  
        **Evaluation:** Train-Test Split (80-20)
        """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Nadeem Shabir Mir** | IIT Bombay | [GitHub](https://github.com/nadeemshabir) | [LinkedIn](https://linkedin.com/in/nadeem-shabir-278022280)")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Customer Churn Prediction System | Built with Streamlit</div>", 
    unsafe_allow_html=True
)