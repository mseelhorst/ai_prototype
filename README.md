# PyTorch Neural Network Example

This repository contains a comprehensive example of creating and training a simple neural network using PyTorch for regression prediction tasks.

## Features

- **Synthetic Data Generation**: Creates a non-linear regression dataset for demonstration
- **Neural Network Architecture**: Simple feedforward network with dropout for regularization
- **Training Loop**: Complete training pipeline with loss tracking and optimization
- **Model Evaluation**: Performance metrics including MSE, RMSE, and R² score
- **Visualization**: Plots of training loss and prediction accuracy
- **Model Persistence**: Save and load trained models
- **GPU Support**: Automatically detects and uses CUDA if available

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- PyTorch (≥2.0.0)
- NumPy (≥1.21.0)
- Matplotlib (≥3.5.0)
- Scikit-learn (≥1.0.0)

## Usage

Simply run the main example file:

```bash
python pytorch_neural_net_example.py
```

## What the Example Does

1. **Data Generation**: Creates 2000 synthetic data points with 3 input features and a non-linear target function: `y = sin(x₁) + cos(x₂) + x₃² + noise`

2. **Model Architecture**: 
   - Input layer: 3 features
   - Hidden layer 1: 64 neurons with ReLU activation
   - Hidden layer 2: 32 neurons with ReLU activation  
   - Output layer: 1 neuron (regression output)
   - Dropout: 20% dropout rate for regularization

3. **Training**: 
   - Optimizer: Adam with learning rate 0.001
   - Loss function: Mean Squared Error (MSE)
   - Batch size: 32
   - Epochs: 100

4. **Evaluation**: 
   - Splits data into 80% training / 20% testing
   - Calculates MSE, RMSE, and R² score
   - Visualizes training loss and prediction accuracy

5. **Model Persistence**: 
   - Saves the trained model, optimizer state, and metrics
   - Demonstrates how to load and use the saved model

## Expected Output

When you run the example, you should see:

```
PyTorch Neural Network Example
========================================

1. Generating synthetic data...
Dataset shape: X=(2000, 3), y=(2000,)

2. Initializing model...
Model architecture:
SimpleNeuralNet(
  (fc1): Linear(in_features=3, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=1, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.2, inplace=False)
)
Total parameters: 2,369

3. Training the model...
Epoch [20/100], Loss: 0.3428
Epoch [40/100], Loss: 0.1245
Epoch [60/100], Loss: 0.0891
Epoch [80/100], Loss: 0.0756
Epoch [100/100], Loss: 0.0698

4. Evaluating the model...

Evaluation Metrics:
Mean Squared Error (MSE): 0.0712
Root Mean Squared Error (RMSE): 0.2669
R² Score: 0.9532

5. Making predictions on new data...

Sample predictions on new data:
Sample 1: Features=[-0.123  0.456  1.789] → Prediction=2.1234
...
```

## Files Generated

After running the example, you'll find:
- `pytorch_model.pth`: Saved model checkpoint
- `pytorch_results.png`: Visualization of training loss and predictions

## Customization

You can easily modify the example:

- **Change the dataset**: Modify `generate_synthetic_data()` function
- **Adjust model architecture**: Edit the `SimpleNeuralNet` class
- **Tune hyperparameters**: Modify learning rate, batch size, epochs, etc.
- **Add more features**: Increase input dimensions and add more layers

## Key Learning Points

This example demonstrates:
- How to define a PyTorch neural network using `nn.Module`
- Proper data handling with `DataLoader`
- Training loop with forward/backward propagation
- Model evaluation and metrics calculation
- Model saving and loading
- GPU utilization when available
- Visualization of results

## Next Steps

To extend this example, consider:
- Adding validation data split for better training monitoring
- Implementing early stopping
- Trying different activation functions
- Adding batch normalization
- Experimenting with different optimizers
- Using real-world datasets
