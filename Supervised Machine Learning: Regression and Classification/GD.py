import numpy as np


def ScaleFeatures(x):
    """
    Perform feature scaling on the input array.

    Args:
        x: Input array, a 2D numpy array where each row represents a training instance and each column represents a feature.

    Returns:
        x_scaled: Scaled array, a 2D numpy array where each element is the scaled value of the corresponding element in the input array.
        mean: Mean values, a 1D numpy array containing the mean values of each feature in the input array.
        stdev: Standard deviation values, a 1D numpy array containing the standard deviation values of each feature in the input array.
    """
    # Calculate mean values along each feature axis
    mean = np.mean(x, axis=0)

    # Calculate standard deviation values along each feature axis
    stdev = np.std(x, axis=0)

    # Scale the input array by subtracting the mean and dividing by the standard deviation
    x_scaled = (x - mean) / (stdev + 1e-8)

    return x_scaled, mean, stdev


def CalculateLoss(x, y, w, b):
    """
    Calculate the loss and cost for a given set of inputs and parameters.

    Args:
    - x: Input features, a 2D numpy array where each row represents a training instance and each column represents a feature.
    - y: Target values, a 1D numpy array where each element represents the target value for a training instance.
    - w: Parameter values, a 1D numpy array representing the weights for each feature.
    - b: Bias value, a scalar representing the bias term.

    Returns:
    - loss: Loss values, a 1D numpy array representing the difference between predicted and actual values.
    - cost: Cost value, a scalar value representing the squared sum of the loss.
    """
    
    # Calculate the loss by performing a dot product of input features with parameters and adding the bias term
    loss = np.dot(x, w) + b - y
    
    # Calculate the cost by summing the squared loss values
    cost = np.sum(loss**2)
    
    return loss, cost



def CalculateGradient(x, y, w, b):
    m = x.shape[0]  # Number of training instances

    # Calculate the loss and cost using the CalculateLoss function
    loss, cost = CalculateLoss(x, y, w, b)

    # Calculate the gradients of the parameter values by performing a dot product of the loss and input features
    dJ_dw = np.dot(loss, x) / m

    # Calculate the gradient of the bias value by summing the loss values
    dJ_db = np.sum(loss) / m

    return dJ_dw, dJ_db, cost



def GD(x, y, iterations = 1000, alpha = 1e-9):
    """
    Performs gradient descent optimization to minimize the cost function and optimize the parameters.

    Args:
    - x: Input features, a 2D numpy array where each row represents a training instance and each column represents a feature.
    - y: Target values, a 1D numpy array where each element represents the target value for a training instance.
    - num_iterations: Number of iterations, an integer value specifying the number of times the gradient descent update will be performed.
    - alpha: Learning rate, a scalar value that determines the step size of the gradient descent update.
    
    Returns:
    - w, b: Learned parameters, a 1D numpy array of size (num_features + 1) containing the optimized parameter values.
    - log: List of cost function values, corresponding gradients and weights with biases

    """

    # Generate initial weights

    w = np.zeros_like(x[0])
    b = 0

    # Initialize log to store steps of the GD
    log = []

    # Perform Gradient Descent
    i = 0
    for i in range(iterations):

        # Calculate gradients
        dJ_dw, dJ_db, cost = CalculateGradient(x, y, w, b)

        # Log the values
        log.append([i, w, b, cost, dJ_dw, dJ_db])

        # adjust the weights and bias
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db

        i += 1

    # Convert list log to numpy
    log = np.array(log, dtype="object")

    return w, b, log