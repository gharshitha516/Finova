import streamlit as st
import pandas as pd
import joblib  # 🔄 Use joblib instead of pickle
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Finova", layout="wide")

st.title("Finova")
st.markdown("Track your expenses, detect fraud, and forecast your budget — all in one dashboard.")

#✅ Load saved model with joblib
@st.cache_resource
def load_model():
    return joblib.load("expense_classifier.pkl")

classifier = load_model()

#Upload CSV
uploaded_file = st.file_uploader("📂 Upload your transaction CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    #✅ Make sure timestamp column is datetime
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    st.subheader("📊 Transaction Overview")
    st.write(df.head())

    #🏷️ Expense Classification
    st.subheader("🏷️ Expense Classification")
    if 'merchant' in df.columns:
        df['merchant'] = df['merchant'].fillna("Unknown")

        try:
            # ✅ Model expects 2D input
            df['category'] = classifier.predict(df[['merchant']])
        except:
            # Fallback if the model expects plain text input
            df['category'] = classifier.predict(df['merchant'])

        fig = px.bar(df['category'].value_counts(), title="Expense Category Distribution")
        st.plotly_chart(fig)
    else:
        st.warning("❗ 'merchant' column not found in uploaded CSV.")

    # 🚨 Fraud Detection
    st.subheader("🚨 Fraudulent Transactions")
    if 'is_fraud' in df.columns:
        frauds = df[df['is_fraud'] == 1]
        st.write(frauds if not frauds.empty else "✅ No fraudulent transactions detected.")
    else:
        st.info("ℹ️ No 'is_fraud' column in data. Skipping fraud detection.")

    # 📈 Budget Forecast with Prophet
    st.subheader("📈 Monthly Budget Forecast")
    if 'amt' in df.columns and 'trans_date_trans_time' in df.columns:
        monthly = df.groupby(pd.Grouper(key='trans_date_trans_time', freq='M'))['amt'].sum().reset_index()
        monthly.columns = ['ds', 'y']

        model = Prophet()
        model.fit(monthly)

        future = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future)

        fig2 = px.line(forecast, x='ds', y='yhat', title='Budget Forecast for Next 6 Months')
        st.plotly_chart(fig2)
    else:
        st.warning("❗ Missing 'amt' or 'trans_date_trans_time' column for forecasting.")
