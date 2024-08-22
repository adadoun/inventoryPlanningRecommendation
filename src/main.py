import argparse

import pandas as pd
from model import SalesNN, ImprovedSalesNN
from predict import get_predictions
from recommend import generate_recommendations

def main(sku):
    # Load test data
    test_data = pd.read_csv('../data/test_data.csv')

    # Filter data for the given SKU
    sku_data = test_data[test_data['SKU'] == sku]

    if sku_data.empty:
        print(f"No data found for SKU: {sku}")
        return

    # Get predictions
    predictions = get_predictions(sku_data,
                                  '../data/NN_sales_prediction_model.pth',
                                  '../data/scaler.save')

    # Generate recommendations
    recommendations = generate_recommendations(predictions)

    # Print results
    print(f"Predictions and Recommendations for SKU: {sku}")
    print("\nPredictions:")
    print(predictions[['DATE', 'QUANTITY_SOLD', 'predictions']])

    print("\nRecommendations:")
    print(recommendations[['Week_Start', 'Reorder_Needed', 'Reorder_Quantity',
                           'Current_Inventory', 'Predicted_Sales']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sales predictions and inventory recommendations for a given SKU")
    parser.add_argument("sku", type=str,
                        help="The SKU to generate predictions and recommendations for")
    args = parser.parse_args()

    main(args.sku)
