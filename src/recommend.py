import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def generate_recommendations(test_data, lead_time=2, safety_stock_factor=1.5):
    """
    Generate weekly inventory recommendations based on sales predictions.

    Args:
        test_data (pd.DataFrame): DataFrame containing test data with actual sales and predictions.
        lead_time (int): Number of weeks it takes for a reorder to arrive. Defaults to 2.
        safety_stock_factor (float): Factor to calculate safety stock. Defaults to 1.5.

    Returns:
        pd.DataFrame: DataFrame containing weekly recommendations for each SKU.
    """
    # Calculate prediction error
    mae = mean_absolute_error(test_data['QUANTITY_SOLD'], test_data['predictions'])

    # Create a DataFrame with actual sales, predictions, and SKUs
    results = pd.DataFrame({
        'SKU': test_data['SKU'],
        'DATE': pd.to_datetime(test_data['DATE']),
        'Actual_Sales': test_data['QUANTITY_SOLD'],
        'Predicted_Sales': test_data['predictions']
    })

    # Get the unique CURRENT_LEVEL for each SKU from the original DataFrame
    current_levels = test_data.groupby('SKU')['CURRENT_LEVEL'].first()

    # Calculate safety stock
    safety_stock = safety_stock_factor * mae * np.sqrt(lead_time)

    # Generate weekly recommendations
    recommendations = []

    # Outer loop: Iterate over each unique SKU
    for sku in results['SKU'].unique():
        sku_data = results[results['SKU'] == sku].sort_values('DATE')
        current_inventory = current_levels.get(sku, 0)

        # Initialize inventory projections for this SKU
        projected_inventory = current_inventory
        projected_inventory_without_reorder = current_inventory
        last_order_week = None

        # Inner loop: Iterate over each week for the current SKU
        for i, row in sku_data.iterrows():
            week_start = row['DATE']
            predicted_sales = row['Predicted_Sales']
            actual_sales = row['Actual_Sales']

            # Check if we need to reorder
            if projected_inventory - predicted_sales <= safety_stock:
                # Reorder is needed
                reorder_quantity = int(predicted_sales * (lead_time + 1) + safety_stock - projected_inventory)
                reorder_quantity = max(reorder_quantity, 0)  # Ensure non-negative quantity

                # Add a recommendation entry with reorder
                recommendations.append({
                    'SKU': sku,
                    'Week_Start': week_start,
                    'Reorder_Needed': 'Yes',
                    'Reorder_Quantity': reorder_quantity,
                    'Current_Inventory': projected_inventory,
                    'Predicted_Sales': predicted_sales,
                    'Actual_Sales': actual_sales,
                    'Projected_Inventory_After_Sales': projected_inventory - predicted_sales,
                    'Projected_Inventory_After_Reorder': projected_inventory - predicted_sales + reorder_quantity,
                    'Projected_Inventory_Without_Reorder': projected_inventory_without_reorder - predicted_sales
                })

                # Update inventory projections considering the reorder
                projected_inventory = projected_inventory - actual_sales + reorder_quantity
                projected_inventory_without_reorder -= actual_sales
                last_order_week = week_start
            else:
                # Reorder is not needed
                recommendations.append({
                    'SKU': sku,
                    'Week_Start': week_start,
                    'Reorder_Needed': 'No',
                    'Reorder_Quantity': 0,
                    'Current_Inventory': projected_inventory,
                    'Predicted_Sales': predicted_sales,
                    'Actual_Sales': actual_sales,
                    'Projected_Inventory_After_Sales': projected_inventory - predicted_sales,
                    'Projected_Inventory_After_Reorder': projected_inventory - predicted_sales,
                    'Projected_Inventory_Without_Reorder': projected_inventory_without_reorder - predicted_sales
                })

                # Update inventory projections without reordering
                projected_inventory -= actual_sales
                projected_inventory_without_reorder -= actual_sales

    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df
