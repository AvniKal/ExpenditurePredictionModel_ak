import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import MonthBegin
import plotly.express as px

st.title("💹 Financial Forecast Dashboard")
st.markdown("This dashboard shows ML-based financial prediction for next fiscal years ")

# --- Load Data ---
df_revenue = pd.read_csv("MAV_Financial Planning Revenue.csv")
df_expenditure = pd.read_csv("MAV_Financial Planning Expenditure.csv")

# --- Data Cleaning ---
def data_clean(df):
    df = df.drop([0, 1])  # remove header rows
    df.columns = ['Cost_Center', 'Account', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df[month_cols] = df[month_cols].apply(pd.to_numeric, errors='coerce')
    return df, month_cols

# --- Forecast Function ---
def run_forecast(df):
    df, month_cols = data_clean(df)
    date_index = pd.date_range('2023-01', periods=12, freq='MS')
    forecast_results = []

    for (cost_center, account), group in df.groupby(['Cost_Center', 'Account']):
        values = group[month_cols].sum().values
        ts = pd.Series(values, index=date_index)

        if len(ts) < 6:
            continue

        try:
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12)
            forecast_index = pd.date_range(ts.index[-1] + MonthBegin(), periods=12, freq='MS')

            forecast_results.append(pd.DataFrame({
                'Date': forecast_index,
                'Forecast': forecast,
                'Cost_Center': cost_center,
                'Account': account
            }))
        except:
            continue

    forecast_df = pd.concat(forecast_results, ignore_index=True)
    cost_centre_forecast = forecast_df.groupby(['Date', 'Cost_Center'])['Forecast'].sum().reset_index()
    total_forecast = forecast_df.groupby('Date')['Forecast'].sum().reset_index()

    return forecast_df, cost_centre_forecast, total_forecast

# --- Tabs for Revenue & Expenditure ---
tab1, tab2 = st.tabs(["📈 Revenue", "📉 Expenditure"])

with tab1:
    st.subheader("Revenue Forecast")
    forecast_df, cost_centre_forecast, total_forecast = run_forecast(df_revenue)

    view_option = st.selectbox("Choose View (Revenue)", ["G/L Account", "Cost Centre", "Total"])
    if view_option == "G/L Account":
        account_list = forecast_df['Account'].unique().tolist()
        selected_account = st.selectbox("Select G/L Account", account_list)
        filtered_data = forecast_df[forecast_df["Account"] == selected_account]
        fig = px.line(filtered_data, x="Date", y="Forecast", title=f"Revenue Forecast for G/L Account: {selected_account}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered_data)

    elif view_option == "Cost Centre":
        cc_list = forecast_df['Cost_Center'].unique().tolist()
        selected_cc = st.selectbox("Select Cost Centre", cc_list)
        filtered_data = forecast_df[forecast_df["Cost_Center"] == selected_cc]
        fig = px.line(filtered_data, x="Date", y="Forecast", title=f"Revenue Forecast for Cost Centre: {selected_cc}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered_data)

    else:
        fig = px.line(total_forecast, x="Date", y="Forecast", title="Total Revenue Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(total_forecast)

with tab2:
    st.subheader("Expenditure Forecast")
    forecast_df, cost_centre_forecast, total_forecast = run_forecast(df_expenditure)

    view_option = st.selectbox("Choose View (Expenditure)", ["G/L Account", "Cost Centre", "Total"])
    if view_option == "G/L Account":
        account_list = forecast_df['Account'].unique().tolist()
        selected_account = st.selectbox("Select G/L Account", account_list)
        filtered_data = forecast_df[forecast_df["Account"] == selected_account]
        fig = px.line(filtered_data, x="Date", y="Forecast", title=f"Expenditure Forecast for G/L Account: {selected_account}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered_data)

    elif view_option == "Cost Centre":
        cc_list = forecast_df['Cost_Center'].unique().tolist()
        selected_cc = st.selectbox("Select Cost Centre", cc_list)
        filtered_data = forecast_df[forecast_df["Cost_Center"] == selected_cc]
        fig = px.line(filtered_data, x="Date", y="Forecast", title=f"Expenditure Forecast for Cost Centre: {selected_cc}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered_data)

    else:
        fig = px.line(total_forecast, x="Date", y="Forecast", title="Total Expenditure Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(total_forecast)
