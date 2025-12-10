# 🔮 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Machine Learning web application** that predicts customer churn probability using Random Forest classification. Built with Streamlit for an interactive user experience, featuring real-time predictions, batch processing, and comprehensive model insights.
🔗 **[Live Demo - Try it here!](https://customer-churn-prediction-by-na.streamlit.app/)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo Screenshots](#-demo-screenshots)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Dataset](#-dataset)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Overview

Customer churn is a critical metric for subscription-based businesses. This project provides an end-to-end solution for:

1. **Predicting** which customers are likely to churn
2. **Identifying** key risk factors driving customer attrition  
3. **Generating** actionable retention recommendations
4. **Visualizing** model performance and business insights

### Business Problem
Companies lose significant revenue when customers leave. Early identification of at-risk customers enables proactive retention strategies, reducing churn rates and increasing customer lifetime value.

### Solution
A machine learning model trained on customer demographics and behavior patterns, deployed as an interactive web application for business users.

---

## ✨ Features

### 🔮 Single Customer Prediction
- Real-time churn probability calculation
- Interactive risk gauge visualization
- Personalized retention recommendations
- Risk factor breakdown analysis

### 📊 Batch Prediction
- Upload CSV files with multiple customers
- Automatic column format detection
- Risk segmentation (Low/Medium/High)
- Downloadable results with predictions
- Visual analytics dashboard

### 📈 Model Insights
- Performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrix visualization
- Feature importance analysis
- Business interpretation of results

### 💡 Smart Recommendations
- Tenure-based loyalty offers
- Price sensitivity interventions
- Age-specific engagement strategies
- Proactive outreach triggers

---

## 🖼️ Demo Screenshots

### Home Dashboard
- Model accuracy metrics at a glance
- Quick navigation to all features
- CSV format guidelines

### Single Prediction
- Input customer details via intuitive controls
- Visual risk gauge (0-100%)
- Color-coded risk levels (🟢 Low, 🟡 Medium, 🔴 High)

### Batch Processing
- Drag-and-drop CSV upload
- Summary statistics cards
- Risk distribution charts
- Segment analysis visualizations

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn, XGBoost |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Model Serialization** | Joblib |

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── 📊 data/
│   └── customer_churn_data.csv    # Training dataset (1000+ records)
│
├── 📓 Notebooks/
│   ├── Notebook.ipynb             # EDA & Model Training
│   ├── model_comparison.png       # Model comparison visualization
│   └── models/
│       ├── model.pkl              # Trained Random Forest model
│       ├── scaler.pkl             # StandardScaler preprocessor
│       └── metrics.json           # Model performance metrics
│
├── 🌐 app.py                      # Streamlit web application
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                   # Project documentation
└── 🔒 .gitignore                  # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nadeemshabir/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## 📖 Usage

### Single Prediction

1. Navigate to **🔮 Single Prediction** from the sidebar
2. Enter customer details:
   - **Gender**: Male/Female
   - **Age**: 18-100 years
   - **Tenure**: 0-72 months
   - **Monthly Charges**: $0-$200
3. Click **"Predict Churn Risk"**
4. View results:
   - Churn probability percentage
   - Risk level classification
   - Personalized recommendations

### Batch Prediction

1. Navigate to **📊 Batch Prediction**
2. Prepare your CSV with columns:
   ```csv
   gender,age,tenure,monthly_charges
   Male,45,12,75.50
   Female,32,48,45.20
   ```
3. Upload the file
4. Click **"Generate Predictions"**
5. Download results with:
   - `churn_probability`
   - `risk_level`
   - `prediction`
   - `recommendation`

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.5% |
| **Precision** | 88.8% |
| **Recall** | 98.3% |
| **F1-Score** | 0.933 |
| **AUC-ROC** | 0.716 |

### Confusion Matrix
The model correctly identifies:
- ✅ True Negatives: Non-churners correctly classified
- ✅ True Positives: Churners correctly identified
- ⚠️ Low false negative rate (high recall) ensures we catch most at-risk customers

### Feature Importance
1. **Tenure** (40%) - Most predictive feature
2. **Monthly Charges** (35%) - Price sensitivity indicator
3. **Age** (18%) - Customer lifecycle stage
4. **Gender** (7%) - Minor demographic factor

---

## 🔌 API Reference

### Input Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `Age` | Integer | Customer age in years | 18-100 |
| `Gender` | String | Customer gender | Male/Female |
| `Tenure` | Integer | Months with company | 0-120 |
| `MonthlyCharges` | Float | Monthly bill amount | 0-200 |

### Output

| Field | Type | Description |
|-------|------|-------------|
| `churn_probability` | Float | Probability of churn (0-1) |
| `risk_level` | String | Low/Medium/High |
| `prediction` | Integer | 0 (No Churn) / 1 (Churn) |
| `recommendation` | String | Suggested action |

---

## 📁 Dataset

### Overview
- **Records**: 1,001 customers
- **Features**: 10 columns
- **Target**: Binary classification (Churn: Yes/No)

### Columns

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique identifier |
| `Age` | Customer age |
| `Gender` | Male/Female |
| `Tenure` | Months as customer |
| `MonthlyCharges` | Monthly bill ($) |
| `ContractType` | Month-to-Month/One-Year/Two-Year |
| `InternetService` | DSL/Fiber Optic/None |
| `TotalCharges` | Cumulative charges |
| `TechSupport` | Yes/No |
| `Churn` | Target variable |

---

## 🔮 Future Improvements

- [ ] Add SHAP values for model explainability
- [ ] Implement real-time data pipeline
- [ ] Add customer segmentation clustering
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add email notification for high-risk customers
- [ ] Implement A/B testing for retention strategies
- [ ] Add more features (payment method, contract details)
- [ ] Create REST API endpoint

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Nadeem Shabir Mir**

- 🎓 IIT Bombay
- 🔗 [GitHub](https://github.com/nadeemshabir)
- 💼 [LinkedIn](https://linkedin.com/in/nadeem-shabir-278022280)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgments

- Dataset inspired by telecom industry churn patterns
- Built with ❤️ using Streamlit and scikit-learn
- Thanks to the open-source community

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>
