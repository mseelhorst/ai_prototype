"""
PyTorch Neural Network Example for Prediction
=============================================

This example demonstrates how to create a simple feedforward neural network
using PyTorch for regression prediction tasks. It includes:
- Synthetic dataset generation
- Neural network model definition
- Training loop with loss calculation
- Model evaluation and prediction
- Visualization of results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SimpleNeuralNet(nn.Module):
    """
    A simple feedforward neural network for regression tasks.
    
    Architecture:
    - Input layer: variable size based on input features
    - Hidden layer 1: 64 neurons with ReLU activation
    - Hidden layer 2: 32 neurons with ReLU activation
    - Output layer: 1 neuron (for regression)
    """
    
    def __init__(self, input_size):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(n_samples=1000, n_features=3, noise_level=0.1):
    """
    Generate synthetic regression data for demonstration.
    
    Args:
        n_samples: Number of data points
        n_features: Number of input features
        noise_level: Amount of noise to add to the target
    
    Returns:
        X: Input features
        y: Target values
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear relationship: y = sin(x1) + cos(x2) + x3^2 + noise
    y = (np.sin(X[:, 0]) + 
         np.cos(X[:, 1]) + 
         X[:, 2]**2 + 
         noise_level * np.random.randn(n_samples))
    
    return X.astype(np.float32), y.astype(np.float32)


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    Train the neural network model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
    
    Returns:
        List of training losses
    """
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions.squeeze(), batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses


def evaluate_model(model, test_loader):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
    
    Returns:
        predictions: Model predictions
        targets: True target values
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_y.cpu().numpy())
    
    return np.array(predictions), np.array(targets)


def plot_results(train_losses, predictions, targets):
    """
    Plot training loss and prediction results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot predictions vs actual
    ax2.scatter(targets, predictions, alpha=0.6)
    ax2.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Predictions vs Actual Values')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/workspace/pytorch_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to demonstrate the complete workflow.
    """
    print("PyTorch Neural Network Example")
    print("=" * 40)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=2000, n_features=3, noise_level=0.1)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # 2. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Initialize model, loss function, and optimizer
    print("\n2. Initializing model...")
    model = SimpleNeuralNet(input_size=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train the model
    print("\n3. Training the model...")
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    # 5. Evaluate the model
    print("\n4. Evaluating the model...")
    predictions, targets = evaluate_model(model, test_loader)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 6. Demonstrate predictions on new data
    print("\n5. Making predictions on new data...")
    
    # Generate some new test samples
    new_X = np.random.randn(5, 3).astype(np.float32)
    new_X_tensor = torch.FloatTensor(new_X).to(device)
    
    model.eval()
    with torch.no_grad():
        new_predictions = model(new_X_tensor).cpu().numpy().flatten()
    
    print("\nSample predictions on new data:")
    for i, (features, pred) in enumerate(zip(new_X, new_predictions)):
        print(f"Sample {i+1}: Features={features} → Prediction={pred:.4f}")
    
    # 7. Plot results
    print("\n6. Plotting results...")
    plot_results(train_losses, predictions, targets)
    
    # 8. Save the trained model
    print("\n7. Saving the model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_metrics': {'mse': mse, 'r2': r2, 'rmse': rmse}
    }, '/workspace/pytorch_model.pth')
    
    print("Model saved as 'pytorch_model.pth'")
    print("\nExample completed successfully!")


def load_and_use_saved_model():
    """
    Example of how to load a saved model and use it for predictions.
    """
    print("\n" + "="*50)
    print("Loading and Using Saved Model")
    print("="*50)
    
    # Load the saved model
    checkpoint = torch.load('/workspace/pytorch_model.pth', map_location=device)
    
    # Recreate the model architecture
    model = SimpleNeuralNet(input_size=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Training metrics from saved model: {checkpoint['test_metrics']}")
    
    # Make predictions with the loaded model
    test_input = torch.FloatTensor([[1.0, -0.5, 2.0]]).to(device)
    
    with torch.no_grad():
        prediction = model(test_input).cpu().numpy()[0][0]
    
    print(f"\nPrediction with loaded model:")
    print(f"Input: [1.0, -0.5, 2.0] → Prediction: {prediction:.4f}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Demonstrate loading and using a saved model
    load_and_use_saved_model()