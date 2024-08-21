import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def generate_recommendations(predictions_df, lead_time=2, safety_stock_factor=1.5):
    # Calculate prediction error
    mae = mean_absolute_error(predictions_df['QUANTITY_SOLD'], predictions_df['predictions'])

    # Create a DataFrame with actual sales, predictions, and SKUs
    results = pd.DataFrame({
        'SKU': predictions_df['SKU'],
        'DATE': pd.to_datetime(predictions_df['DATE']),
        'Actual_Sales': predictions_df['QUANTITY_SOLD'],
        'Predicted_Sales': predictions_df['predictions']
    })

    # Get the unique CURRENT_LEVEL for each SKU
    current_levels = predictions_df.groupby('SKU')['CURRENT_LEVEL'].first()

    # Calculate safety stock
    safety_stock = safety_stock_factor * mae * np.sqrt(lead_time)

    # Generate recommendations
    recommendations = []

    for sku in results['SKU'].unique():
        sku_data = results[results['SKU'] == sku].sort_values('DATE')
        current_inventory = current_levels.get(sku, 0)

        # Calculate projected inventory for each week
        projected_inventory = current_inventory
        projected_inventory_without_reorder = current_inventory

        for i, row in sku_data.iterrows():
            week_start = row['DATE']
            predicted_sales = row['Predicted_Sales']
            actual_sales = row['Actual_Sales']

            # Check if we need to reorder
            if projected_inventory - predicted_sales <= safety_stock:
                # Calculate reorder quantity using predicted sales
                reorder_quantity = int(predicted_sales * (lead_time + 1) + safety_stock - projected_inventory)
                reorder_quantity = max(reorder_quantity, 0)  # Ensure non-negative quantity

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

                projected_inventory = projected_inventory - actual_sales + reorder_quantity
                projected_inventory_without_reorder -= actual_sales
            else:
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

                projected_inventory -= actual_sales
                projected_inventory_without_reorder -= actual_sales

    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df
