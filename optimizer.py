import numpy as np

def clip_gradients_by_norm(gradients, max_norm=1.0):
    """
    Clips gradients by their norm to prevent exploding gradients.

    Parameters:
    ----------
    gradients : dict
        Dictionary containing the gradients for each parameter. 
        Keys are strings representing the gradient names (e.g., "dW1", "db1").
    max_norm : float, optional
        The maximum norm value to which the gradients should be clipped. 
        Default is 1.0.

    Returns:
    -------
    gradients : dict
        Dictionary containing the clipped gradients. 
        Gradients are scaled if their norm exceeds the `max_norm`.
    """
    total_norm = np.sqrt(sum(np.sum(np.square(grad)) for grad in gradients.values()))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients.values():
            grad *= clip_coef
    return gradients

def SGD(parameters, grads, num_layers, learning_rate=0.001):
    """
        Implements the Stochastic Gradient Descent (SGD) update rule.
    
    Parameters:
    ----------
    parameters : dict
        Dictionary containing the parameters W and b of each layer.
    grads : dict
        Dictionary containing the gradients dW and db for each layer.
    num_layers : int
        Number of layers in the neural network.
    learning_rate : float, optional
        Learning rate for the parameter update, default is 0.001.

    Returns:
    -------
    parameters : dict
        Updated parameters after the SGD step.
    """
    for i in range(1, num_layers + 1):
        parameters['W'+str(i)] = parameters['W'+str(i)] - learning_rate * grads['dW'+str(i)]
        parameters['b'+str(i)] = parameters['b'+str(i)] - learning_rate * grads['db'+str(i)]
    return parameters

def initialize_momentum(parameters, num_layers):
    """
    Initializes the moment estimates for the momentum optimizer.

    Parameters:
    ----------
    parameters : dict
        Dictionary containing the weights and biases of the model.
    num_layers : int
        The number of layers in the model.

    Returns:
    -------
    v : dict
        Dictionary containing the initialized moment estimates for the weights and biases.
        Keys are "dW{l}" and "db{l}" where l is the layer index.
    """
    v = {}
    for i in range(1, num_layers + 1):
        v["dW" + str(i)] = np.zeros_like(parameters["W" + str(i)])
        v["db" + str(i)] = np.zeros_like(parameters["b" + str(i)])
    return v

def gradient_descent_with_momentum(parameters, grads, num_layers, v, learning_rate=0.01, beta=0.9):
    """
    Implements gradient descent with momentum.
    
    Parameters:
    ----------
    parameters : dict
        Dictionary containing the parameters W and b of each layer.
    grads : dict
        Dictionary containing the gradients dW and db for each layer.
    num_layers : int
        Number of layers in the neural network.
    v : dict
        Dictionary containing the moving averages of the gradients (velocity terms).
    learning_rate : float, optional
        Learning rate for the parameter update, default is 0.01.
    beta : float, optional
        Momentum hyperparameter, default is 0.9.

    Returns:
    -------
    parameters : dict
        Updated parameters after the gradient descent with momentum step.
    v : dict
        Updated velocity terms.
    """
    for i in range(1, num_layers + 1):
        v["dW" + str(i)] = beta * v["dW" + str(i)] + (1 - beta) * grads['dW' + str(i)]
        v["db" + str(i)] = beta * v["db" + str(i)] + (1 - beta) * grads['db' + str(i)]

        parameters["W" + str(i)] -= learning_rate * v["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * v["db" + str(i)]
    
    return parameters, v

def initialize_adam(parameters, num_layers):
    """
    Initializes the moment estimates for the Adam optimizer.

    Parameters:
    ----------
    parameters : dict
        Dictionary containing the weights and biases of the model.
    num_layers : int
        The number of layers in the model.

    Returns:
    -------
    v : dict
        Dictionary containing the initialized moment estimates for the weights and biases.
        Keys are "dW{l}" and "db{l}" where l is the layer index.
    s : dict
        Dictionary containing the initialized squared gradient estimates for the weights and biases.
        Keys are "dW{l}" and "db{l}" where l is the layer index.
    """
    v = {}
    s = {}
    for l in range(1, num_layers + 1):
        v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])

    return v, s

def adam(parameters, grads, num_layers, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implements the Adam optimization algorithm.
    
    Parameters:
    ----------
    parameters : dict
        Dictionary containing the parameters W and b of each layer.
    grads : dict
        Dictionary containing the gradients dW and db for each layer.
    num_layers : int
        Number of layers in the neural network.
    v : dict
        Dictionary containing the first moment estimates (moving averages of the gradients).
    s : dict
        Dictionary containing the second moment estimates (moving averages of the squared gradients).
    t : int
        Iteration number.
    learning_rate : float, optional
        Learning rate for the parameter update, default is 0.001.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates, default is 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates, default is 0.999.
    epsilon : float, optional
        Small constant for numerical stability, default is 1e-8.

    Returns:
    -------
    parameters : dict
        Updated parameters after the Adam step.
    v : dict
        Updated first moment estimates.
    s : dict
        Updated second moment estimates.
    """
    v_corrected = {}
    s_corrected = {}

    for i in range(1, num_layers + 1):
        v["dW" + str(i)] = beta1 * v["dW" + str(i)] + (1 - beta1) * grads['dW' + str(i)]
        v["db" + str(i)] = beta1 * v["db" + str(i)] + (1 - beta1) * grads['db' + str(i)]

        v_corrected["dW" + str(i)] = v["dW" + str(i)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(i)] = v["db" + str(i)] / (1 - np.power(beta1, t))

        s["dW" + str(i)] = beta2 * s["dW" + str(i)] + (1 - beta2) * np.power(grads['dW' + str(i)], 2)
        s["db" + str(i)] = beta2 * s["db" + str(i)] + (1 - beta2) * np.power(grads['db' + str(i)], 2)

        s_corrected["dW" + str(i)] = s["dW" + str(i)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(i)] = s["db" + str(i)] / (1 - np.power(beta2, t))

        parameters["W" + str(i)] -= learning_rate * v_corrected["dW" + str(i)] / (np.sqrt(s_corrected["dW" + str(i)]) + epsilon)
        parameters["b" + str(i)] -= learning_rate * v_corrected["db" + str(i)] / (np.sqrt(s_corrected["db" + str(i)]) + epsilon)

    return parameters, v, s