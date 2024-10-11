import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
import matplotlib.pyplot as plt

class OrderBookDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        return (self.data[index:index+self.sequence_length, :-1], self.data[index+self.sequence_length, -1])

def preprocess_data(data):
    scaler = MinMaxScaler()
    data[['bid_price', 'ask_price', 'bid_volume', 'ask_volume']] = scaler.fit_transform(
        data[['bid_price', 'ask_price', 'bid_volume', 'ask_volume']]
    )
    return data

def create_sequences(data, sequence_length):
    dataset = OrderBookDataset(data, sequence_length)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
def evaluate_model(model, data_loader):
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    return np.concatenate(predictions), np.concatenate(targets)

if __name__ == "__main__":
    # Load and preprocess the data
    data = pd.read_csv('synthetic_order_book.csv')
    data = preprocess_data(data)
    
    # Prepare training data
    sequence_length = 20
    order_book_values = data[['bid_price', 'ask_price', 'bid_volume', 'ask_volume']].values
    train_loader = create_sequences(order_book_values, sequence_length)
    
    # Initialize the model
    input_size = 4
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    
    # Train the model
    train_model(model, train_loader, num_epochs=10, learning_rate=0.001)
    
    # Evaluate the model
    predictions, targets = evaluate_model(model, train_loader)
    
    # Plot the results
    plt.plot(predictions, label="Predicted Prices")
    plt.plot(targets, label="Actual Prices")
    plt.legend()
    plt.show()
