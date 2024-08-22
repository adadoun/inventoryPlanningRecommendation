import streamlit as st
import pandas as pd
import os
import sys
from src.predict import get_predictions
from src.recommend import generate_recommendations
from src.model import SalesNN
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


def load_data():
    data = pd.read_csv('data/test_data.csv')
    # Convert DATE column to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])
    return data

def get_unique_skus(data):
    return sorted(data['SKU'].unique())

def plot_sales_and_inventory(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=data['Week_Start'], y=data['Actual_Sales'], name="Actual Sales"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data['Week_Start'], y=data['Predicted_Sales'], name="Predicted Sales"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data['Week_Start'], y=data['Current_Inventory'], name="Current Inventory"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=data['Week_Start'], y=data['Projected_Inventory_After_Sales'], name="Projected Inventory"),
        secondary_y=True,
    )

    reorder_points = data[data['Reorder_Needed'] == 'Yes']
    fig.add_trace(
        go.Scatter(x=reorder_points['Week_Start'], y=reorder_points['Reorder_Quantity'],
                   mode='markers', name='Reorder Quantity', marker=dict(size=10, color='red')),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Sales, Inventory, and Reorder Points",
        xaxis_title="Date",
    )

    fig.update_yaxes(title_text="Sales", secondary_y=False)
    fig.update_yaxes(title_text="Inventory / Reorder Quantity", secondary_y=True)

    return fig

def main():
    st.title("Sales Prediction and Inventory Recommendation")

    # Load data
    data = load_data()
    skus = get_unique_skus(data)

    # Sidebar
    st.sidebar.header("Settings")
    selected_sku = st.sidebar.selectbox("Select SKU", skus)

    # Filter data for the selected SKU
    sku_data = data[data['SKU'] == selected_sku]

    if sku_data.empty:
        st.write(f"No data found for SKU: {selected_sku}")
        return

    # Get predictions
    predictions = get_predictions(sku_data,
                                  'data/NN_sales_prediction_model.pth',
                                  'data/scaler.save')

    # Generate recommendations
    recommendations = generate_recommendations(predictions)

    # Ensure 'Week_Start' is datetime
    recommendations['Week_Start'] = pd.to_datetime(recommendations['Week_Start'])

    # Display recommendations
    st.header("Recommendations")
    st.dataframe(recommendations[['Week_Start', 'Reorder_Needed', 'Reorder_Quantity',
                                  'Current_Inventory', 'Predicted_Sales']])

    # Plot
    #fig = plot_sales_and_inventory(recommendations)
    #st.plotly_chart(fig)

    # Time slider
    st.header("Time Navigation")
    min_date = recommendations['Week_Start'].min().date()
    max_date = recommendations['Week_Start'].max().date()
    selected_date = st.slider("Select Date", min_value=min_date, max_value=max_date, value=max_date, format="YYYY-MM-DD")

    # Convert selected_date back to datetime for filtering
    selected_datetime = datetime.combine(selected_date, datetime.min.time())

    # Filter data based on selected date
    filtered_data = recommendations[recommendations['Week_Start'] <= selected_datetime]

    # Update plot with filtered data
    filtered_fig = plot_sales_and_inventory(filtered_data)
    st.plotly_chart(filtered_fig)

    # Display filtered recommendations
    st.header("Filtered Recommendations")
    st.dataframe(filtered_data[['Week_Start', 'Reorder_Needed', 'Reorder_Quantity',
                                'Current_Inventory', 'Predicted_Sales']])

if __name__ == "__main__":
    main()
