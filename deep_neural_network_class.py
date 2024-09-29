import numpy as np
from activation import *
from optimizer import *
from loss import *

class Deep_Neural_Network:
    """
    A class to represent a multi-layer neural network.
    
    Attributes:
    ----------
    layers : list
        A list containing the dimensions of each layer in the neural network.
    num_layers : int
        The number of layers in the network, excluding the input layer.
    parameters : dict
        A dictionary storing the parameters (weights and biases) for each layer.
    """

    info = {
        'activations': ('relu', 'sigmoid'),
        'optimizers': ('SGD', 'momentum', 'adam'),
        'loss': ('MSE', 'BinaryCrossentropy', 'CategoricalCrossentropy'),
        'loss_to_activation': {
            'MSE': 'linear',
            'CategoricalCrossentropy': 'softmax', 
            'BinaryCrossentropy': 'sigmoid'
        }
    }

    def __init__(self, dims):
        """
        Initializes the NeuralNetwork object with layer dimensions and number of layers.
        
        Parameters:
        ----------
        dims : list
            A list where each element represents the number of units in each layer 
            of the network (including input and output layers).
        """
        assert len(dims) > 2, "Network must have at least 3 layers (input, hidden, output)"
        self.layers = dims
        self.num_layers = len(dims) - 1
        self.parameters = {}
        self.initialize_params(dims)

    def initialize_params(self, dims):
        """
        Initializes the parameters (weights and biases) for each layer in the network.
        Weights are initialized using He initialization, and biases are initialized to zeros.
        """
        for l in range(1, self.num_layers + 1):
            self.parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2. / dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((dims[l], 1))

    def compile(self, optimizer='SGD', activation='relu', loss='BinaryCrossentropy'):
        """
        Configures the model with selected optimizer, activation function, and loss function.
        Also initializes necessary parameters based on the optimizer type.
        """
        self.optimizer = optimizer if optimizer in self.info['optimizers'] else 'SGD'
        self.activation = activation if activation in self.info['activations'] else 'relu'
        self.loss_func = loss if loss in self.info['loss'] else 'BinaryCrossentropy'
        self.output_activation = self.info['loss_to_activation'].get(self.loss_func, 'sigmoid')
        
        if optimizer == 'adam':
            self.v, self.s = initialize_adam(self.parameters, self.num_layers)
            self.t = 0
        elif optimizer == 'momentum':
            self.v = initialize_momentum(self.parameters, self.num_layers)

    def train_test_split(self, X, Y, validation_split=0.2, shuffle=True):
        """
        Splits the data into training and validation sets.
        """
        m = X.shape[1]
        if shuffle:
            permutation = np.random.permutation(m)
            X = X[:, permutation]
            Y = Y[:, permutation]

        split_idx = int(m * (1 - validation_split))
        X_train, X_val = X[:, :split_idx], X[:, split_idx:]
        Y_train, Y_val = Y[:, :split_idx], Y[:, split_idx:]

        return X_train, X_val, Y_train, Y_val

    def compute_cost(self, AL, Y, lambda_reg=0.01):
        """
        Computes the total cost including the regularization term.
        """
        m = Y.shape[1]
        base_loss = Loss(self.loss_func).loss(AL, Y, m)

        l2_cost = 0
        for l in range(1, self.num_layers + 1):
            l2_cost += np.sum(np.square(self.parameters[f'W{l}']))
        l2_cost *= (lambda_reg / (2 * m))

        cost = base_loss + l2_cost
        return np.squeeze(cost)

    def step_forward(self, A_prev, W, b, activation):
        """
        Implements the forward propagation for a single layer.
        """
        Z = np.dot(W, A_prev) + b
        
        if activation == "linear":
            A = Z
        elif activation == "sigmoid":
            A = sigmoid(Z)
        else:
            A = relu(Z)
        
        cache = (A_prev, W, Z)
        return A, cache

    def forward(self, X):
        """
        Implements the full forward propagation through the network.
        """
        caches = []
        A = X

        for l in range(1, self.num_layers + 1):
            A_prev = A
            activation = self.output_activation if l == self.num_layers else self.activation
            A, cache = self.step_forward(A_prev, self.parameters[f'W{l}'], self.parameters[f'b{l}'], activation)
            caches.append(cache)
        return A, caches
    
    def step_backward(self, dA, cache, activation):
        """
        Implements the backward propagation for a single layer.
        """
        A_prev, W, Z = cache
        m = A_prev.shape[1]

        if activation == "linear":
            dZ = dA
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        else:
            dZ = relu_backward(dA, Z)

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches, epsilon, lambda_reg=0.01):
        """
        Implements the full backward propagation through the network.
        """
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = Derivative(self.output_activation, m=m, epsilon=epsilon).dAL(AL, Y)

        current_cache = caches[self.num_layers - 1]
        grads[f"dA{self.num_layers-1}"], grads[f"dW{self.num_layers}"], grads[f"db{self.num_layers}"] = \
            self.step_backward(dAL, current_cache, self.output_activation)
        
        grads[f"dW{self.num_layers}"] += (lambda_reg / m) * self.parameters[f'W{self.num_layers}']

        for l in reversed(range(self.num_layers - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.step_backward(grads[f"dA{l+1}"], current_cache, self.activation)
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp + (lambda_reg / m) * self.parameters[f'W{l+1}']
            grads[f"db{l+1}"] = db_temp

        return grads
    
    def update_parameters(self, grads, current_learning_rate, beta1, beta2, epsilon):
        """
        Updates parameters using the selected optimizer.
        """
        if self.optimizer == 'SGD':
            self.parameters = SGD(self.parameters, grads, self.num_layers, current_learning_rate)
        elif self.optimizer == 'momentum':
            self.parameters, self.v = gradient_descent_with_momentum(self.parameters, grads, self.num_layers, self.v, current_learning_rate, beta=beta1)
        elif self.optimizer == 'adam':
            self.t += 1
            self.parameters, self.v, self.s = adam(self.parameters, grads, self.num_layers, self.v, self.s, self.t, current_learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

    def fit(self, X, Y, learning_rate=0.001, epoch=1000, batch_size=64, verbose=False, validation_split=0.2,
            shuffle=True, lambda_reg=0.01, decay_rate=0.95, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Trains the neural network on the given dataset.
        """
        if X.shape[0] != self.layers[0]:
            X = X.T
            Y = Y.reshape(1, -1)
        assert X.shape[0] == self.layers[0], f"Input shape {X.shape[0]} does not match first layer size {self.layers[0]}"

        X_train, X_val, Y_train, Y_val = self.train_test_split(X, Y, validation_split, shuffle)

        m = X_train.shape[1]
        costs = []
        val_costs = []
        num_batches = int(np.ceil(m / batch_size))

        for i in range(epoch):
            for j in range(num_batches):
                start = j * batch_size
                end = min((j + 1) * batch_size, m)
                X_batch = X_train[:, start:end]
                Y_batch = Y_train[:, start:end]

                AL, caches = self.forward(X_batch)
                grads = self.backward(AL, Y_batch, caches, epsilon, lambda_reg)

                current_learning_rate = learning_rate * (decay_rate ** (i // 100))
                self.update_parameters(grads, current_learning_rate, beta1, beta2, epsilon)

            train_AL, _ = self.forward(X_train)
            train_cost = self.compute_cost(train_AL, Y_train, lambda_reg)
            costs.append(train_cost)

            val_AL, _ = self.forward(X_val)
            val_cost = self.compute_cost(val_AL, Y_val, lambda_reg)
            val_costs.append(val_cost)

            if verbose and (i % 100 == 0 or i == epoch - 1):
                print(f"Epoch {i+1}/{epoch} --> Train cost: {train_cost:.8f}, Validation cost: {val_cost:.8f}")

            if i > 0 and abs(costs[-1] - costs[-2]) < 1e-8:
                print("Early stopping due to negligible improvement")
                break

        self.evaluate(X_train, Y_train, "Training", reshape=False)
        self.evaluate(X_val, Y_val, "Validation", reshape=False)
        return costs, val_costs

    def evaluate(self, X, Y, dataset_name="", reshape=True, batch_size=64, compute_cost=False):
        """
        Evaluates the model's performance on a given dataset.
        """
        if reshape:
            X = X.T
            Y = Y.reshape(1, -1)
        
        m = X.shape[1]
        num_batches = int(np.ceil(m / batch_size))
        all_predictions = []
        total_cost = 0
        
        for j in range(num_batches):
            start = j * batch_size
            end = min((j + 1) * batch_size, m)
            X_batch = X[:, start:end]
            Y_batch = Y[:, start:end]
            
            AL, _ = self.forward(X_batch)
            
            if compute_cost:
                total_cost += self.compute_cost(AL, Y_batch) * (end - start)
            
            if self.output_activation == 'sigmoid':
                pred = (AL > 0.5).astype(int)
            elif self.output_activation == 'softmax':
                pred = np.argmax(AL, axis=0)
            else:
                pred = AL
            
            all_predictions.extend(pred.flatten())
        
        all_predictions = np.array(all_predictions)
        Y = Y.flatten()
        
        if self.output_activation in ['sigmoid', 'softmax']:
            accuracy = np.mean(all_predictions == Y)
            if dataset_name:
                print(f"{dataset_name} Accuracy: {accuracy:.2%}")
        else:
            mse = np.mean((all_predictions - Y) ** 2)
            if dataset_name:
                print(f"{dataset_name} MSE: {mse:.4f}")
        
        if compute_cost:
            avg_cost = total_cost / m
            return avg_cost
        else:
            return all_predictions

    def predict(self, X):
        """
        Predicts the class labels for the input data `X` based on the trained model.
        """
        probas, _ = self.forward(X)

        if self.output_activation == 'sigmoid':
            return (probas > 0.5).astype(int)
        elif self.output_activation == 'softmax':
            return np.argmax(probas, axis=0)
        else:
            return probas  # For regression tasks
