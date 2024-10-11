# Market Microstructure Modeling with LSTM

This project predicts short-term price movements using LSTM based on synthetic order book data.

## Files
- `main.py`: Main script for training and evaluating the LSTM model.
- `data_generator.py`: Script to generate synthetic order book data.
- `model.py`: Contains the PyTorch LSTM model definition.
- `synthetic_order_book.csv`: Generated synthetic order book data.

## Requirements
- Python 3.x
- PyTorch
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## How to Run

1. Install the required packages:

    ```
    pip install torch numpy pandas matplotlib scikit-learn
    ```

2. Generate synthetic data:

    ```
    python data_generator.py
    ```

3. Train the LSTM model:

    ```
    python main.py
    ```

4. View the prediction plot.
