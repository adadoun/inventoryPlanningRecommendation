# Sales Prediction and Inventory Recommendation System

This project implements a sales prediction and inventory recommendation system using neural networks. It predicts future sales for given SKUs and provides inventory reorder recommendations based on these predictions.

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
