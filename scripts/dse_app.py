"""
dse_app.py — DSE Project Streamlit Dashboard
Run: streamlit run scripts/dse_app.py
Opens at: http://localhost:8501
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DSE — MF Transaction Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

RAW  = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\raw')
PROC = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\processed')

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=80)
st.sidebar.title("DSE Project")
st.sidebar.markdown("**Mutual Fund Transaction Prediction**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📋 Problem Definition",
    "📊 Dataset Description",
    "🔍 EDA",
    "🤖 Algorithm Comparison",
    "📈 Performance Analysis",
    "🏆 Key Findings",
    "🚀 Future Scope",
    "🎯 Live Predictor"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Algorithms Used:**")
st.sidebar.markdown("- Logistic Regression (ML)")
st.sidebar.markdown("- Decision Tree (ML)")
st.sidebar.markdown("- Random Forest (ML)")
st.sidebar.markdown("- KNN (ML)")
st.sidebar.markdown("- Neural Network (DL)")

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(RAW / '08_investor_transactions.csv')
    return df

@st.cache_resource
def train_models(df):
    df2 = df.copy()
    features = ['age_group','gender','city_tier','annual_income_lakh',
                'amount_inr','payment_mode','kyc_status']
    target = 'transaction_type'
    df2 = df2[features + [target]].dropna()

    le = LabelEncoder()
    encoders = {}
    for col in ['age_group','gender','city_tier','payment_mode','kyc_status']:
        df2[col] = le.fit_transform(df2[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df2[target] = le_target.fit_transform(df2[target])

    X = df2[features].values
    y = df2[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Neural Network (DL)': MLPClassifier(hidden_layer_sizes=(128,64,32),
                                   activation='relu', max_iter=500,
                                   random_state=42, early_stopping=True),
    }

    results = []
    trained = {}
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        results.append({
            'Algorithm': name,
            'Accuracy':  round(acc*100, 2),
            'Precision': round(prec*100, 2),
            'Recall':    round(rec*100, 2),
            'F1-Score':  round(f1*100, 2),
            'MAE':       round(mae, 4),
            'MSE':       round(mse, 4),
            'RMSE':      round(rmse, 4),
            'R2':        round(r2, 4),
        })
        trained[name] = (model, y_pred)

    return pd.DataFrame(results), trained, X_test, y_test, le_target, scaler, encoders, features

df = load_data()

# ── HOME ──────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📊 Data Science Essentials Project")
    st.subheader("Mutual Fund Investor Transaction Type Prediction")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Features",        "13")
    col3.metric("Target Classes",  "3")
    col4.metric("Algorithms",      "5 (4ML + 1DL)")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Project Overview
        - **Dataset:** Investor Transactions (AMFI India)
        - **Task:** Multi-class Classification
        - **Target:** Predict SIP / Lumpsum / Redemption
        - **Algorithms:** LR, DT, RF, KNN, Neural Network
        """)
    with col2:
        st.markdown("""
        ### All 8 Components Covered
        1. ✅ Problem Definition
        2. ✅ Expected Output
        3. ✅ Dataset Description
        4. ✅ EDA (6 Charts)
        5. ✅ Algorithm Comparison (5 Algos)
        6. ✅ Performance Analysis (8 Metrics)
        7. ✅ Key Findings
        8. ✅ Future Scope
        """)

# ── PROBLEM DEFINITION ────────────────────────────────────────
elif page == "📋 Problem Definition":
    st.title("📋 Component 1 & 2 — Problem Definition & Expected Output")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Problem Definition
        Indian mutual fund investors make 3 types of transactions:
        - **SIP** — Systematic Investment Plan
        - **Lumpsum** — One-time investment
        - **Redemption** — Withdrawal

        **Goal:** Predict transaction type from investor profile

        **Significance:**
        - Predict SIP discontinuation early
        - Enable retention campaigns
        - Personalised investor communication
        - Improve B30 city penetration strategy
        """)
    with col2:
        st.markdown("""
        ### Expected Output
        1. Trained classification model
        2. Accuracy, Precision, Recall, F1 for 5 algorithms
        3. Confusion matrix for best algorithm
        4. Feature importance ranking
        5. Algorithm comparison table
        6. Best algorithm recommendation

        ### ML Task
        - **Type:** Multi-class Classification
        - **Input:** Demographics + Financial profile
        - **Output:** SIP / Lumpsum / Redemption
        """)

# ── DATASET DESCRIPTION ───────────────────────────────────────
elif page == "📊 Dataset Description":
    st.title("📊 Component 3 — Dataset Description")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", df.shape[1])
    col3.metric("Target Classes", df['transaction_type'].nunique())

    st.markdown("### Column Details")
    desc = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.astype(str).values,
        'Unique Values': [df[c].nunique() for c in df.columns],
        'Null Count': [df[c].isna().sum() for c in df.columns],
        'Sample': [str(df[c].iloc[0]) for c in df.columns]
    })
    st.dataframe(desc, use_container_width=True)

    st.markdown("### Target Distribution")
    tx_cnt = df['transaction_type'].value_counts().reset_index()
    tx_cnt.columns = ['Transaction Type', 'Count']
    tx_cnt['Percentage'] = (tx_cnt['Count'] / len(df) * 100).round(2)
    st.dataframe(tx_cnt, use_container_width=True)

    st.markdown("### Preprocessing Steps")
    st.markdown("""
    1. **Label Encoding** — age_group, gender, city_tier, payment_mode, kyc_status
    2. **StandardScaler** — Normalize all features (required for KNN and Neural Network)
    3. **Train-Test Split** — 80% train, 20% test, stratified by target
    4. **Random State = 42** — Reproducibility
    """)

    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ── EDA ───────────────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("🔍 Component 4 — Exploratory Data Analysis")
    st.markdown("---")
    colors = ['#42A5F5','#66BB6A','#EF5350']

    # Chart 1
    st.subheader("Chart 1 — Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    tx_cnt = df['transaction_type'].value_counts()
    bars = ax.bar(tx_cnt.index, tx_cnt.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, tx_cnt.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Transaction Type Distribution', fontweight='bold')
    st.pyplot(fig); plt.close()

    # Chart 2 & 3
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chart 2 — Age Group")
        fig, ax = plt.subplots(figsize=(7,4))
        age_tx = df.groupby(['age_group','transaction_type']).size().unstack(fill_value=0)
        age_order = [a for a in ['18-25','26-35','36-45','46-55','56+'] if a in age_tx.index]
        age_tx.reindex(age_order).plot(kind='bar', ax=ax, color=colors, edgecolor='white')
        ax.set_title('Age Group vs Transaction Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig); plt.close()
    with col2:
        st.subheader("Chart 3 — Amount Distribution")
        fig, ax = plt.subplots(figsize=(7,4))
        for tx, color in zip(df['transaction_type'].unique(), colors):
            ax.hist(df[df['transaction_type']==tx]['amount_inr'], bins=20, alpha=0.6, label=tx, color=color)
        ax.set_title('Amount Distribution by Type', fontweight='bold')
        ax.legend(); ax.set_xlim(0, 110000)
        st.pyplot(fig); plt.close()

    # Chart 4 & 5
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chart 4 — City Tier")
        fig, ax = plt.subplots(figsize=(7,4))
        tier_tx = df.groupby(['city_tier','transaction_type']).size().unstack(fill_value=0)
        tier_tx.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
        ax.set_title('City Tier vs Transaction Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig); plt.close()
    with col2:
        st.subheader("Chart 5 — Gender")
        fig, ax = plt.subplots(figsize=(7,4))
        gen_tx = df.groupby(['gender','transaction_type']).size().unstack(fill_value=0)
        gen_tx.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
        ax.set_title('Gender vs Transaction Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig); plt.close()

    # Chart 6
    st.subheader("Chart 6 — Income vs Transaction Type")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.boxplot([df[df['transaction_type']==t]['annual_income_lakh'].values
                for t in df['transaction_type'].unique()],
               labels=df['transaction_type'].unique(),
               patch_artist=True,
               boxprops=dict(facecolor='#E3F2FD'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title('Annual Income vs Transaction Type', fontweight='bold')
    st.pyplot(fig); plt.close()

# ── ALGORITHM COMPARISON ──────────────────────────────────────
elif page == "🤖 Algorithm Comparison":
    st.title("🤖 Component 5 — Algorithm Comparison")
    st.markdown("---")

    st.markdown("""
    | # | Algorithm | Type | Description |
    |---|---|---|---|
    | 1 | Logistic Regression | ML | Linear classifier, baseline model |
    | 2 | Decision Tree | ML | Tree-based, interpretable model |
    | 3 | Random Forest | ML | Ensemble of 100 decision trees |
    | 4 | KNN | ML | Instance-based, k=5 neighbors |
    | 5 | Neural Network (MLP) | **DL** | 3 hidden layers: 128→64→32, ReLU |
    """)

    with st.spinner("Training all 5 algorithms... please wait"):
        results_df, trained_models, X_test, y_test, le_target, scaler, encoders, features = train_models(df)

    st.success("✅ All 5 algorithms trained successfully!")
    st.dataframe(results_df, use_container_width=True)

# ── PERFORMANCE ANALYSIS ──────────────────────────────────────
elif page == "📈 Performance Analysis":
    st.title("📈 Component 6 — Performance Analysis")
    st.markdown("---")

    with st.spinner("Training models..."):
        results_df, trained_models, X_test, y_test, le_target, scaler, encoders, features = train_models(df)

    best = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(f"🏆 Best Algorithm: **{best['Algorithm']}** with **{best['Accuracy']}% Accuracy**")

    # Metrics table
    st.subheader("All Metrics Table")
    st.dataframe(results_df.set_index('Algorithm'), use_container_width=True)

    algo_colors = ['#42A5F5','#66BB6A','#EF5350','#FFA726','#AB47BC']

    # Chart 1: Accuracy
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(results_df['Algorithm'], results_df['Accuracy'], color=algo_colors, edgecolor='white')
    for bar, val in zip(bars, results_df['Accuracy']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,115)
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig); plt.close()

    # Chart 2 & 3
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precision / Recall / F1")
        fig, ax = plt.subplots(figsize=(8,5))
        x = np.arange(len(results_df)); w = 0.25
        ax.bar(x,     results_df['Precision'], w, label='Precision', color='#42A5F5', edgecolor='white')
        ax.bar(x+w,   results_df['Recall'],    w, label='Recall',    color='#66BB6A', edgecolor='white')
        ax.bar(x+2*w, results_df['F1-Score'],  w, label='F1-Score',  color='#EF5350', edgecolor='white')
        ax.set_xticks(x+w); ax.set_xticklabels(results_df['Algorithm'], rotation=15)
        ax.legend(); ax.set_ylabel('Score (%)')
        st.pyplot(fig); plt.close()
    with col2:
        st.subheader("Confusion Matrix")
        best_name = best['Algorithm']
        _, best_pred = trained_models[best_name]
        cm = confusion_matrix(y_test, best_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_, ax=ax)
        ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        st.pyplot(fig); plt.close()

    # Feature importance
    st.subheader("Feature Importance — Random Forest")
    rf_model = trained_models['Random Forest'][0]
    feat_imp = pd.DataFrame({'Feature':features, 'Importance':rf_model.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='Blues_r', ax=ax)
    ax.set_title('Feature Importance', fontweight='bold')
    st.pyplot(fig); plt.close()

    # Classification report
    st.subheader(f"Classification Report — {best_name}")
    report = classification_report(y_test, best_pred, target_names=le_target.classes_)
    st.text(report)

# ── KEY FINDINGS ──────────────────────────────────────────────
elif page == "🏆 Key Findings":
    st.title("🏆 Component 7 — Key Findings & Conclusion")
    st.markdown("---")

    with st.spinner("Loading results..."):
        results_df, trained_models, X_test, y_test, le_target, scaler, encoders, features = train_models(df)

    best = results_df.loc[results_df['Accuracy'].idxmax()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Algorithm", best['Algorithm'])
    col2.metric("Best Accuracy",  f"{best['Accuracy']}%")
    col3.metric("Best F1-Score",  f"{best['F1-Score']}%")
    col4.metric("Total Algorithms", "5")

    st.markdown("### Algorithm Ranking")
    ranking = results_df[['Algorithm','Accuracy','Precision','Recall','F1-Score']].sort_values('Accuracy', ascending=False)
    st.dataframe(ranking, use_container_width=True)

    st.markdown("### Conclusion")
    st.info(f"""
    **{best['Algorithm']}** performed best overall with **{best['Accuracy']}% accuracy**.

    - Random Forest: Best due to ensemble approach handling non-linear patterns
    - Neural Network (DL): Competitive performance with deep feature learning
    - SIP is easiest to predict — highest frequency class
    - Redemption is hardest — class imbalance challenge
    - amount_inr and annual_income_lakh are top predictive features
    """)

# ── FUTURE SCOPE ──────────────────────────────────────────────
elif page == "🚀 Future Scope":
    st.title("🚀 Component 8 — Future Scope")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Advanced ML/DL
        - **LSTM/RNN** — Sequential SIP pattern prediction
        - **XGBoost** — Better class imbalance handling
        - **SMOTE** — Oversample Redemption class
        - **AutoML** — Automated hyperparameter tuning

        ### Deployment
        - **FastAPI** — REST API for real-time prediction
        - **Docker** — Containerized deployment
        - **Cloud** — AWS/GCP deployment
        """)
    with col2:
        st.markdown("""
        ### Explainability
        - **SHAP Values** — Model interpretability
        - **LIME** — Local explanations for advisors
        - **Feature Drift** — Monitor data changes

        ### Data Enhancement
        - **Real-time AMFI feed** — Live transaction data
        - **NAV performance** — Add as features
        - **Market VIX** — Volatility as external factor
        - **State GDP** — Geographic economic indicators
        """)

# ── LIVE PREDICTOR ────────────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.title("🎯 Live Transaction Type Predictor")
    st.markdown("---")
    st.info("Enter investor details to predict transaction type using all 5 algorithms!")

    with st.spinner("Training models..."):
        results_df, trained_models, X_test, y_test, le_target, scaler, encoders, features = train_models(df)

    col1, col2 = st.columns(2)
    with col1:
        age_group       = st.selectbox("Age Group", ['18-25','26-35','36-45','46-55','56+'])
        gender          = st.selectbox("Gender", ['Male','Female'])
        city_tier       = st.selectbox("City Tier", ['T30','B30'])
        annual_income   = st.slider("Annual Income (₹ Lakh)", 1.0, 50.0, 10.0, 0.5)
    with col2:
        amount          = st.selectbox("Transaction Amount (₹)", [500,1000,2000,5000,10000,25000,50000,100000])
        payment_mode    = st.selectbox("Payment Mode", ['UPI','Net Banking','Mandate','Cheque'])
        kyc_status      = st.selectbox("KYC Status", ['Verified','Pending'])

    if st.button("🔮 Predict Transaction Type", type="primary"):
        # Encode input
        input_data = pd.DataFrame([{
            'age_group': age_group, 'gender': gender,
            'city_tier': city_tier, 'annual_income_lakh': annual_income,
            'amount_inr': amount, 'payment_mode': payment_mode,
            'kyc_status': kyc_status
        }])

        for col in ['age_group','gender','city_tier','payment_mode','kyc_status']:
            le_temp = LabelEncoder()
            le_temp.fit(df[col].astype(str))
            try:
                input_data[col] = le_temp.transform(input_data[col].astype(str))
            except:
                input_data[col] = 0

        X_input = scaler.transform(input_data[features].values)

        st.markdown("### Predictions from all 5 algorithms:")
        pred_cols = st.columns(5)
        algo_colors_map = {
            'Logistic Regression': '🔵',
            'Decision Tree':       '🟢',
            'Random Forest':       '🔴',
            'KNN':                 '🟠',
            'Neural Network (DL)': '🟣'
        }
        for i, (name, (model, _)) in enumerate(trained_models.items()):
            pred = model.predict(X_input)[0]
            label = le_target.classes_[pred]
            prob = model.predict_proba(X_input)[0].max() * 100
            with pred_cols[i]:
                st.metric(
                    f"{algo_colors_map[name]} {name}",
                    label,
                    f"{prob:.1f}% confidence"
                )

        # Majority vote
        preds = [le_target.classes_[m.predict(X_input)[0]] for m, _ in trained_models.values()]
        from collections import Counter
        final = Counter(preds).most_common(1)[0][0]
        st.success(f"### 🏆 Final Prediction (Majority Vote): **{final}**")
