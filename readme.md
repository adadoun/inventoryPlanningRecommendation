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
├── notebooks/
│   ├── ArimaSalesPrediction.ipynb
│   ├── DataPreparation.ipynb
│   ├── EDASalesData.ipynb
│   └── EDAStepSizeOptimization.ipynb
│   ├── LgbmSalesPrediction.ipynb
│   ├── NNSalesPrediction.ipynb
│   └── PlanningRecommendation.ipynb 
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── recommend.py
├── recommenderApp.py
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

## Components

### Data

- `train_data.csv` and `test_data.csv`: Training and testing datasets
- `scaler.save`: Saved StandardScaler object for feature scaling
- `NN_sales_prediction_model.pth`: Saved neural network model

### Notebooks

- `ArimaSalesPrediction.ipynb`: ARIMA model development
- `DataPreparation.ipynb`: Data preprocessing and feature engineering
- `EDASalesData.ipynb`: Exploratory Data Analysis of sales data
- `EDAStepSizeOptimization.ipynb`: Analysis for optimizing time step size
- `LgbmSalesPrediction.ipynb`: LightGBM model development
- `NNSalesPrediction.ipynb`: Neural Network model development
- `PlanningRecommendation.ipynb`: Inventory planning recommendations development

### Source Code

- `data_preparation.py`: Functions for data preprocessing and feature engineering
- `model.py`: Neural network model architecture
- `train.py`: Script for training the neural network model
- `predict.py`: Functions for making predictions
- `recommend.py`: Functions for generating inventory recommendations

## Recommender App

The `recommenderApp.py` file contains a Streamlit-based user interface for interacting with the sales prediction and inventory management system.

### How to Launch the App

Run the Streamlit app:
``` streamlit run recommenderApp.py ```

The app will open in your default web browser.

### Features and Interaction

- **SKU Selection**: Choose a specific SKU from the dropdown menu.
- **Time Navigation**: Use the time slider to view predictions and recommendations for different time periods.
- **Visualization**: 
- View interactive plots showing actual sales, predicted sales, current inventory, and projected inventory levels.
- Reorder points are highlighted on the plot.
- **Recommendations**: 
- See detailed recommendations for reordering, including when to reorder and suggested quantities.
- View current inventory levels and predicted sales.

The app provides a user-friendly interface for inventory managers to make data-driven decisions about stock levels and reordering for individual SKUs.

## Performance

The neural network model's performance is evaluated using metrics such as MAE, RMSE, and MAPE. The system demonstrates a high stockout prevention rate of 94.75%, significantly reducing potential stockouts from 2325 to 122.

For detailed performance analysis and further information about the project, please refer to the individual notebook files in the `notebooks/` directory.

## Future Work

- Implement SKU segmentation for more tailored predictions
- Incorporate external factors (e.g., promotions, economic indicators) into the model
- Develop a system for continuous model retraining and performance monitoring
- Integrate inventory cost optimization to balance holding costs and stockout risks
