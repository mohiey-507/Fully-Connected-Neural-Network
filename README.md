# Fully-Connected Neural Network

Building a Deep Fully-Connected(Dense) from scratch

A flexible and modular implementation of deep neural networks in Python, supporting both classification and regression tasks with various optimizers, activation functions, and loss functions.

## Features

- Multiple layer support with configurable dimensions
- Various optimization algorithms:
  - Stochastic Gradient Descent (SGD)
  - Momentum-based gradient descent
  - Adam optimizer
- Multiple activation functions:
  - ReLU
  - Sigmoid
  - Linear
  - Softmax (for output layer)
- Different loss functions:
  - Mean Squared Error (MSE)
  - Binary Cross-entropy
  - Categorical Cross-entropy
- Training features:
  - Mini-batch gradient descent
  - L2 regularization
  - Learning rate decay
  - Early stopping
  - Training/validation split
  - Batch normalization

## Project Structure

```
.
├── deep_neural_network_class.py  # Main neural network implementation
├── activation.py                 # Activation functions and their derivatives
├── loss.py                      # Loss function implementations
├── optimizer.py                 # Optimization algorithms
└── main.py                     # Example usage and benchmarking
```

## Requirements

- NumPy
- scikit-learn (for example datasets and comparison)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [repository-url]
cd [repository-name]
pip install numpy
```

## Usage

### Basic Example

```python
from deep_neural_network_class import Deep_Neural_Network
import numpy as np

# Create a neural network with 3 layers: input(4), hidden(8), output(1)
model = Deep_Neural_Network([4, 8, 1])

# Configure the model
model.compile(
    optimizer='adam',
    activation='relu',
    loss='BinaryCrossentropy'
)

# Train the model
costs, val_costs = model.fit(
    X_train,
    y_train,
    learning_rate=0.01,
    epoch=1000,
    batch_size=32,
    validation_split=0.2
)

# Make predictions
predictions = model.predict(X_test)
```

### Example Tasks

The repository includes example implementations for both classification and regression tasks in `main.py`:

1. Classification Task:
   - Uses the breast cancer dataset from scikit-learn
   - Compares performance with LogisticRegression and MLPClassifier

2. Regression Task:
   - Creates a synthetic dataset using sine and cosine functions
   - Compares performance with MLPRegressor

To run the examples:

```bash
python main.py
```

## Model Configuration Options

### Optimizers
- `'SGD'`: Standard stochastic gradient descent
- `'momentum'`: Momentum-based gradient descent
- `'adam'`: Adam optimizer

### Activation Functions
- `'relu'`: Rectified Linear Unit
- `'sigmoid'`: Sigmoid function
- `'linear'`: Linear activation (for regression tasks)

### Loss Functions
- `'MSE'`: Mean Squared Error (for regression)
- `'BinaryCrossentropy'`: Binary Cross-entropy (for binary classification)
- `'CategoricalCrossentropy'`: Categorical Cross-entropy (for multi-class classification)

## Advanced Features

### Early Stopping
```python
model.fit(X, y, early_stopping_patience=50)
```

### Learning Rate Decay
```python
model.fit(X, y, learning_rate=0.01, decay_rate=0.95)
```

### L2 Regularization
```python
model.fit(X, y, lambda_reg=0.01)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
