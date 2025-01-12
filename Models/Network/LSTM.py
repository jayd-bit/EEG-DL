import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess EEG dataset
def load_and_preprocess_data():
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])  # Motor Imagery dataset (Competition IV 2a)

    # Preprocess: band-pass filter, standardization
    preprocessors = [
        Preprocessor("filterbank", freq_bands=[(4, 40)]),  # Band-pass filter 4-40Hz
        Preprocessor("zscore")  # Standardization
    ]
    preprocess(dataset, preprocessors)

    # Create epochs and split dataset
    windows_dataset = create_windows_from_events(dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0)
    X, y = [], []
    for window in windows_dataset:
        X.append(window[0])
        y.append(window[1])

    # Convert to numpy arrays and split into train-test sets
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Define xLSTM model
class xLSTM(nn.Module):
    def __init__(self, input_size, lstm_size, num_classes):
        super(xLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_size, lstm_size // 2)
        self.fc2 = nn.Linear(lstm_size // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(lstm_size // 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.softplus(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.softmax(x, dim=1)

# Define training and evaluation functions
def train_model(model, dataloader, optimizer, criterion, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize models
    models = {
        "xLSTM": xLSTM(input_size, lstm_size=128, num_classes=num_classes).to(device),
        "LSTM": nn.LSTM(input_size, 128, batch_first=True).to(device),
        "BiLSTM": nn.LSTM(input_size, 128, batch_first=True, bidirectional=True).to(device)
    }

    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        train_model(model, train_loader, optimizer, criterion, num_epochs=20)

        print(f"Evaluating {model_name}...")
        accuracy = evaluate_model(model, test_loader)
        results[model_name] = accuracy
        print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Print results
    print("\nPerformance Comparison:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
