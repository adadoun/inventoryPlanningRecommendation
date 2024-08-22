# Sales Prediction and Inventory Recommendation System

This project implements a sales prediction and inventory recommendation system using neural networks. 
It predicts future sales for given SKUs and provides inventory reorder recommendations based on these predictions.

The reordering feature serves several important purposes in the inventory management:

- Error Calculation:
  * It calculates the Mean Absolute Error (MAE) of the sales predictions, which is used in safety stock calculations.

- Safety Stock Calculation:
  * It computes a safety stock level based on the prediction error, lead time, and a safety factor.
  * This helps to buffer against uncertainties in demand and lead time.

- Inventory Projection:
  * For each SKU, it projects future inventory levels based on current inventory, predicted sales, and actual sales.
  * It maintains two projections: one with reordering and one without, to show the impact of the reordering strategy.

- Reorder Decision Making:
  * It determines when to place a reorder based on whether the projected inventory falls below the safety stock level.
  * When a reorder is needed, it calculates the reorder quantity based on predicted sales and lead time.

- Comprehensive Recommendation Generation:
  * For each week and SKU, it generates a detailed recommendation including:
      1. Whether a reorder is needed
      2. The recommended reorder quantity
      3. Current inventory levels
      4. Predicted and actual sales
      5. Projected inventory levels under different scenarios

The main objective of this function is to provide a data-driven approach to inventory management.
It aims to:
* Maintain sufficient stock to meet demand (avoiding stockouts)
* Avoid excessive inventory (reducing holding costs)
* Provide visibility into future inventory levels
* Demonstrate the impact of the reordering strategy compared to no reordering

## Project Structure

```
sales-prediction-project/
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── scaler.save
│   └── NN_sales_prediction_model.pth
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── recommend.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sales-prediction-project.git
   cd sales-prediction-project
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate predictions and recommendations for a specific SKU, run:

```
python main.py <SKU>
```

Replace `<SKU>` with the actual SKU you want to analyze.

## Training the Model

If you want to retrain the model with new data, you can modify the `src/train.py` file and run it separately.

## Data

Make sure to place your training and test data in the `data/` directory. The model and scaler files should also be in this directory after training.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs.

## License

This project is licensed under the MIT License.
