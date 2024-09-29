import numpy as np 

class Loss:
    def __init__(self, loss_func):
        """
        Initializes the loss function object based on the input type.

        Parameters:
        ----------
        loss_func : str
            The name of the loss function ('MSE', 'BinaryCrossentropy', or 'CategoricalCrossentropy').
        """
        loss_functions = {
            'MSE': self.MSE,
            'CategoricalCrossentropy': self.CategoricalCrossentropy,
            'BinaryCrossentropy': self.BinaryCrossentropy
        }
        # Set the selected loss function or default to BinaryCrossentropy
        self.loss = loss_functions.get(loss_func, self.BinaryCrossentropy)

    def MSE(self, AL, Y, m):
        """
        Mean Squared Error (MSE) Loss.
        
        Parameters:
        ----------
        AL : numpy array
            The predicted outputs (activations) from the model.
        Y : numpy array
            The true labels.
        m : int
            Number of examples.

        Returns:
        -------
        float
            The computed MSE loss.
        """
        return (1 / (2 * m)) * np.sum(np.square(AL - Y))

    def BinaryCrossentropy(self, AL, Y, m):
        """
        Binary Crossentropy Loss.

        Parameters:
        ----------
        AL : numpy array
            The predicted outputs (activations) from the model.
        Y : numpy array
            The true binary labels (0 or 1).
        m : int
            Number of examples.

        Returns:
        -------
        float
            The computed binary crossentropy loss.
        """
        AL = np.clip(AL, 1e-8, 1 - 1e-8)  # To avoid log(0) errors
        return (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

    def CategoricalCrossentropy(self, AL, Y, m):
        """
        Categorical Crossentropy Loss.

        Parameters:
        ----------
        AL : numpy array
            The predicted outputs (activations) from the model.
        Y : numpy array
            The true one-hot encoded labels.
        m : int
            Number of examples.

        Returns:
        -------
        float
            The computed categorical crossentropy loss.
        """
        AL = np.clip(AL, 1e-8, 1 - 1e-8)  # To avoid log(0) errors
        return (-1 / m) * np.sum(Y * np.log(AL))
