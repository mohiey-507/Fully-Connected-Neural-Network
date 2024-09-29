
import numpy as np

def relu(Z):
    """
    Applies the ReLU (Rectified Linear Unit) activation function.

    Parameters:
    ----------
    Z : numpy.ndarray
        Input array or matrix on which the ReLU activation function is applied.

    Returns:
    -------
    A : numpy.ndarray
        Output array or matrix after applying the ReLU activation function. 
        The output has the same shape as the input.
    """
    A = np.maximum(0, Z)
    return A

def sigmoid(Z):
    """
    Applies the sigmoid activation function.

    Parameters:
    ----------
    Z : numpy.ndarray
        Input array or matrix on which the sigmoid activation function is applied.

    Returns:
    -------
    A : numpy.ndarray
        Output array or matrix after applying the sigmoid activation function. 
        The output has the same shape as the input.
    """
    A = 1 / (1 + np.exp(-Z))
    return A


def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, Z):
    """
    Backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """            
    a = 1/(1+np.exp(-Z))
    dZ = dA * a * (1-a)

    return dZ

class Derivative(object):
    """
    A class to compute the derivative of the loss function with respect to the activation function.

    Attributes:
    ----------
    dAL : function
        Function to compute the derivative of the loss with respect to the activation function.
    epsilon : float
        A small value to avoid division by zero in calculations. Default is 1e-8.
    m : int, optional
        Number of examples in the dataset. Used in the linear loss function. Default is None.

    Methods:
    -------
    sigmoid(AL, Y)
        Computes the derivative of the binary cross-entropy loss with respect to the sigmoid activation.
    linear(AL, Y)
        Computes the derivative of the mean squared error loss with respect to the linear activation.
    softmax(AL, Y)
        Computes the derivative of the categorical cross-entropy loss with respect to the softmax activation.
    """

    def __init__(self, activation, m=None, epsilon=1e-8):
        """
        Initializes the Derivative class.

        Parameters:
        ----------
        activation : str
            The activation function used for the loss function. 
            Possible values are 'sigmoid', 'linear', and 'softmax'.
        m : int, optional
            Number of examples in the dataset. Used in the linear loss function. Default is None.
        epsilon : float, optional
            A small value to avoid division by zero in calculations. Default is 1e-8.
        """
        derivatives = {
            'sigmoid': self.sigmoid,
            'linear': self.linear,
            'softmax': self.softmax}

        self.dAL = derivatives.get(activation, self.sigmoid)
        self.epsilon = epsilon
        self.m = m

    def sigmoid(self, AL, Y):
        """
        Computes the derivative of the binary cross-entropy loss with respect to the sigmoid activation.

        Parameters:
        ----------
        AL : numpy.ndarray
            The predicted output values from the sigmoid activation function.
        Y : numpy.ndarray
            The true labels.

        Returns:
        -------
        numpy.ndarray
            The gradient of the loss function with respect to the sigmoid activation.
        """
        return -(np.divide(Y, AL + self.epsilon) - np.divide(1 - Y, 1 - AL + self.epsilon))
    
    def linear(self, AL, Y):
        """
        Computes the derivative of the mean squared error loss with respect to the linear activation.

        Parameters:
        ----------
        AL : numpy.ndarray
            The predicted output values from the linear activation function.
        Y : numpy.ndarray
            The true labels.

        Returns:
        -------
        numpy.ndarray
            The gradient of the loss function with respect to the linear activation.
        """
        return (AL - Y) / self.m
    
    def softmax(self, AL, Y):
        """
        Computes the derivative of the categorical cross-entropy loss with respect to the softmax activation.

        Parameters:
        ----------
        AL : numpy.ndarray
            The predicted output values from the softmax activation function.
        Y : numpy.ndarray
            The true labels.

        Returns:
        -------
        numpy.ndarray
            The gradient of the loss function with respect to the softmax activation.
        """
        return AL - Y
