import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
import mne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class xLSTMCell(nn.Module):
    """Implementation of xLSTM cell with extended gating mechanisms"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Extended gates
        self.xgate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Traditional LSTM gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, 
                hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Extended gating mechanism
        x_gate = torch.sigmoid(self.xgate(combined))
        
        # Traditional LSTM gates
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        
        # Modified cell state update with extended gating
        c_t = f_t * c_prev + i_t * c_tilde * x_gate
        
        # Hidden state update
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class BaseModel(nn.Module):
    """Base class for EEG classification models"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))

class StandardLSTM(BaseModel):
    """Standard LSTM implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(StandardLSTM, self).__init__(input_size, hidden_size, num_classes)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class BiLSTM(BaseModel):
    """Bidirectional LSTM implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(BiLSTM, self).__init__(input_size, hidden_size, num_classes)
        self.lstm = nn.LSTM(input_size, hidden_size // 2, 
                           batch_first=True, bidirectional=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class xLSTM(BaseModel):
    """Extended LSTM implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(xLSTM, self).__init__(input_size, hidden_size, num_classes)
        self.xlstm_cell = xLSTMCell(input_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        hidden = self.init_hidden(batch_size, device)
        outputs = []
        
        for t in range(seq_len):
            hidden = self.xlstm_cell(x[:, t, :], hidden)
            outputs.append(hidden[0])
        
        out = self.fc(outputs[-1])
        return out

def load_bci_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load BCI Competition IV 2a dataset"""
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[1])
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=400,
        window_stride_samples=200,
        preload=True
    )
    
    X = windows_dataset.get_data()
    y = windows_dataset.get_labels()
    
    return X, y

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                valid_loader: DataLoader,
                device: torch.device,
                num_epochs: int = 100) -> Dict[str, list]:
    """Train the model and return training history"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_true = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        # Update history
        train_acc = accuracy_score(train_true, train_preds)
        val_acc = accuracy_score(val_true, val_preds)
        
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Train Loss: {np.mean(train_losses):.4f}, '
                       f'Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {np.mean(val_losses):.4f}, '
                       f'Val Acc: {val_acc:.4f}')
    
    return history

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load and preprocess data
    X, y = load_bci_data()
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Create dataloaders
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    input_size = X.shape[2]  # Number of EEG channels
    hidden_size = 128
    num_classes = len(torch.unique(y))
    
    # Initialize models
    models = {
        'StandardLSTM': StandardLSTM(input_size, hidden_size, num_classes),
        'BiLSTM': BiLSTM(input_size, hidden_size, num_classes),
        'xLSTM': xLSTM(input_size, hidden_size, num_classes)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        logger.info(f'\nTraining {name}...')
        model = model.to(device)
        history = train_model(model, train_loader, valid_loader, device)
        results[name] = history
        
        # Print final results
        logger.info(f'\n{name} Final Results:')
        logger.info(f'Final Train Accuracy: {history["train_acc"][-1]:.4f}')
        logger.info(f'Final Validation Accuracy: {history["val_acc"][-1]:.4f}')

if __name__ == "__main__":
    main()
