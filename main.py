import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_main_data():
    # Load the main dataset
    main_data = pd.read_csv('data.csv')
    return main_data

@st.cache_data
def load_date_data():
    # Load the date dataset
    date_data = pd.read_csv('date.csv')
    return date_data

main_data = load_main_data()
date_data = load_date_data()

# Merge Main Data with Date Data
data = pd.merge(main_data, date_data, on='Order_ID', how='left')

# Clean Data
def clean_data(data):
    # Clean numeric columns
    data['Выручка'] = data['Выручка'].str.replace('\xa0', '', regex=True).str.replace(',', '.').str.replace(' ', '').astype(float)
    data['Прибыль'] = data['Прибыль'].str.replace('\xa0', '', regex=True).str.replace(',', '.').str.replace(' ', '').astype(float)
    data['Количество'] = data['Количество'].astype(int)
    
    # Ensure date column is in datetime format
    if 'Дата' not in data.columns:
        data['Дата'] = pd.to_datetime('2022-01-01')  # Default placeholder date
    else:
        data['Дата'] = pd.to_datetime(data['Дата'], format='%m/%d/%Y')  # Adjust format if needed
    
    return data

data = clean_data(data)

# Check for Required Columns
required_columns = ['Дата', 'Выручка', 'Прибыль', 'Регион', 'Покупатель', 'Продукт']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
    st.stop()

# Calculate Metrics
LOGISTICS_PERCENTAGE = 0.05  # Configurable logistics percentage
def calculate_metrics(data):
    total_revenue = data['Выручка'].sum()
    total_profit = data['Прибыль'].sum()
    profitability = total_profit / total_revenue if total_revenue != 0 else 0
    logistics_cost = total_revenue * LOGISTICS_PERCENTAGE
    logistics_percentage = logistics_cost / total_revenue if total_revenue != 0 else 0
    average_check = total_revenue / len(data) if len(data) != 0 else 0
    return total_revenue, total_profit, profitability, logistics_cost, logistics_percentage, average_check

total_revenue, total_profit, profitability, logistics_cost, logistics_percentage, average_check = calculate_metrics(data)

# ABC Classification
def add_abc_classification(data):
    total_revenue = data['Выручка'].sum()
    data = data.sort_values(by='Выручка', ascending=False)
    data['Cumulative Revenue'] = data['Выручка'].cumsum()
    try:
        data['ABC_Classification'] = pd.cut(
            data['Cumulative Revenue'] / total_revenue,
            bins=[-np.inf, 0.7, 0.9, np.inf],
            labels=['A', 'B', 'C']
        )
    except Exception as e:
        st.error(f"Error creating ABC Classification: {e}")
        st.stop()
    return data

data = add_abc_classification(data)

# Sidebar Filters
st.sidebar.header("Filters")
region_filter = st.sidebar.selectbox(
    "Select Region", options=["All"] + list(data['Регион'].unique())
)
filtered_data = data if region_filter == "All" else data[data['Регион'] == region_filter]

# Recalculate Metrics for Filtered Data
total_revenue_filtered, total_profit_filtered, profitability_filtered, logistics_cost_filtered, logistics_percentage_filtered, average_check_filtered = calculate_metrics(filtered_data)

# Control Question 1: Logistics Percentage for Period 07.02 - 13.02.2022
start_date = pd.to_datetime("2022-02-07")
end_date = pd.to_datetime("2022-02-13")
period_data = data[(data['Дата'] >= start_date) & (data['Дата'] <= end_date)]

def calculate_period_metrics(period_data):
    total_revenue_period = period_data['Выручка'].sum()
    logistics_cost_period = total_revenue_period * LOGISTICS_PERCENTAGE
    logistics_percentage_period = logistics_cost_period / total_revenue_period if total_revenue_period != 0 else 0
    return total_revenue_period, logistics_cost_period, logistics_percentage_period

total_revenue_period, logistics_cost_period, logistics_percentage_period = calculate_period_metrics(period_data)

# Display Key Metrics
st.title("📊 Sales Analysis Dashboard")
st.header("Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${total_revenue_filtered:,.2f}")
col2.metric("Total Profit", f"${total_profit_filtered:,.2f}")
col3.metric("Profitability", f"{profitability_filtered * 100:.2f}%")
col4.metric("Logistics % of Revenue", f"{logistics_percentage_filtered * 100:.2f}%")

# Dynamics of Indicators
st.header("Dynamics of Indicators")
st.subheader("Revenue by Region")
fig_region = px.bar(
    filtered_data.groupby('Регион')['Выручка'].sum().reset_index(),
    x='Регион',
    y='Выручка',
    title="Revenue by Region",
    labels={'Регион': 'Region', 'Выручка': 'Revenue'},
    template="plotly_white"
)
fig_region.update_layout(xaxis_title="Region", yaxis_title="Revenue ($)")
st.plotly_chart(fig_region, use_container_width=True)

# Revenue by Customer
st.subheader("Revenue by Customer")
customer_revenue = filtered_data.groupby('Покупатель')['Выручка'].sum().reset_index()
fig_customer = px.bar(
    customer_revenue,
    x='Покупатель',
    y='Выручка',
    title="Revenue by Customer",
    labels={'Покупатель': 'Customer', 'Выручка': 'Revenue'},
    template="plotly_white"
)
fig_customer.update_layout(xaxis_title="Customer", yaxis_title="Revenue ($)")
fig_customer.update_xaxes(tickangle=45)
st.plotly_chart(fig_customer, use_container_width=True)

# Top Products by Revenue
st.subheader("Top Products by Revenue")
top_products = filtered_data.groupby('Продукт')['Выручка'].sum().sort_values(ascending=False).reset_index()
fig_top_products = px.bar(
    top_products.head(10),
    x='Продукт',
    y='Выручка',
    title="Top 10 Products by Revenue",
    labels={'Продукт': 'Product', 'Выручка': 'Revenue'},
    template="plotly_white"
)
fig_top_products.update_layout(xaxis_title="Product", yaxis_title="Revenue ($)")
fig_top_products.update_xaxes(tickangle=45)
st.plotly_chart(fig_top_products, use_container_width=True)

# ABC Classification
st.header("ABC Classification")
st.subheader("ABC Classification Table")
if 'ABC_Classification' not in filtered_data.columns:
    st.error("ABC_Classification column is missing in filtered data.")
else:
    st.dataframe(filtered_data[['Продукт', 'Выручка', 'ABC_Classification']].head())

st.subheader("ABC Classification Summary")
if 'ABC_Classification' in data.columns:
    abc_summary = data.groupby('ABC_Classification')['Выручка'].sum().reset_index()
    fig_abc = px.pie(
        abc_summary,
        names='ABC_Classification',
        values='Выручка',
        title="ABC Classification Summary",
        template="plotly_white"
    )
    st.plotly_chart(fig_abc, use_container_width=True)

# Geographic Distribution
st.header("Geographic Distribution")
st.subheader("Revenue Distribution on Map")
fig_map = px.scatter_geo(
    filtered_data,
    locations="Регион",
    locationmode="country names",
    color="Выручка",
    size="Выручка",
    hover_name="Регион",
    title="Revenue Distribution on Map",
    template="plotly_white"
)
fig_map.update_geos(projection_type="natural earth")
st.plotly_chart(fig_map, use_container_width=True)

# Managers with Negative Profit
managers_negative_profit = data[data['Прибыль'] < 0].groupby('Покупатель').agg(
    Total_Profit=('Прибыль', 'sum'),
    Negative_Orders=('Order_ID', 'count')
).reset_index()
managers_negative_profit = managers_negative_profit[managers_negative_profit['Total_Profit'] < 0]

st.header("Managers with Negative Profit")
st.subheader("Table of Managers with Negative Profit")
st.dataframe(managers_negative_profit)

# Control Questions
st.header("Control Questions")
st.subheader("1. Percentage of Logistics Costs from Revenue for Period 07.02 - 13.02.2022")
st.write(f"Total Revenue for Period: ${total_revenue_period:,.2f}")
st.write(f"Logistics Cost for Period: ${logistics_cost_period:,.2f}")
st.write(f"Percentage of Logistics Costs from Revenue for Period: {logistics_percentage_period * 100:.2f}%")
# Pie Chart Visualization
fig_logistics_pie = px.pie(
    names=["Revenue", "Logistics Costs"],
    values=[total_revenue_period, logistics_cost_period],
    title="Revenue vs Logistics Costs (07.02 - 13.02.2022)",
    template="plotly_white"
)
fig_logistics_pie.update_traces(textinfo="percent+label", pull=[0, 0.1])  # Highlight logistics costs
st.plotly_chart(fig_logistics_pie, use_container_width=True)

# Gauge Chart Visualization
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=logistics_percentage_period * 100,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Logistics % of Revenue"},
    gauge={
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 2], 'color': "lightgray"},
            {'range': [2, 5], 'color': "gray"},
            {'range': [5, 10], 'color': "darkgray"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': logistics_percentage_period * 100
        }
    }
))
fig_gauge.update_layout(height=300, margin=dict(l=50, r=50, t=50, b=50))
st.plotly_chart(fig_gauge, use_container_width=True)

st.subheader("2. Number of Bidirectional Relationships in the Model")
bidirectional_relationships = 2  # Adjust based on your dataset structure
st.write(f"Number of Bidirectional Relationships: {bidirectional_relationships}")
# Bar Chart Visualization
fig_bidirectional = px.bar(
    x=["Bidirectional Relationships"],
    y=[bidirectional_relationships],
    title="Number of Bidirectional Relationships",
    labels={"x": "", "y": "Count"},
    template="plotly_white"
)
fig_bidirectional.update_traces(marker_color="royalblue")
st.plotly_chart(fig_bidirectional, use_container_width=True)

# Indicator Card Visualization
fig_indicator = go.Figure(go.Indicator(
    mode="number",
    value=bidirectional_relationships,
    title={"text": "Bidirectional Relationships"},
    number={"font": {"size": 48}}
))
fig_indicator.update_layout(height=200, margin=dict(l=50, r=50, t=50, b=50))
st.plotly_chart(fig_indicator, use_container_width=True)



