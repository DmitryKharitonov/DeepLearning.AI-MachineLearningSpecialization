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
    # Calculate mean and standard deviation values along each feature axis
    mean = np.mean(x, axis=0)
    stdev = np.std(x, axis=0)

    # Scale the input array by subtracting the mean and dividing by the standard deviation
    x_scaled = (x - mean) / (stdev + 1e-8)

    return x_scaled, mean, stdev


def CalculateLoss_Classification(x, y, w, b, lambda_ = 0):
    """
    Calculates the loss and cost using the provided sigmoid function.

    Args:
        x: Input values.
        y: Target values.
        w: Weight values.
        b: Bias value.
        lambda_: Regularization parameter.

    Returns:
        Tuple: loss and cost values.
    """

    m = x.shape[0]

    sigmoid = lambda x, w, b : 1 / (1 + np.exp(-1 * (np.dot(x, w) + b) ) )
    
    upper_half = - np.log(sigmoid(x, w, b))
    lower_half = - np.log(1 - sigmoid(x, w, b))

    loss = sigmoid(x, w, b) - y

    cost = np.dot(y, upper_half) + np.dot((1 - y),lower_half)
    cost = np.sum(cost) / m

    regularization = lambda_ * np.sum(w**2) / 2 / m

    cost_reg = cost + regularization

    return loss, cost_reg


def CalculateLoss_Regression(x, y, w, b, lambda_ = 0):
    """
    Calculate the loss and cost for a given set of inputs and parameters.

    Args:
    x: Input features, a 2D numpy array where each row represents a training instance
       and each column represents a feature.
    y: Target values, a 1D numpy array where each element represents the target value
       for a training instance.
    w: Parameter values, a 1D numpy array representing the weights for each feature.
    b: Bias value, a scalar representing the bias term.
    lambda_: Regularization parameter.

    Returns:
    loss: Loss values, a 1D numpy array representing the difference between predicted
          and actual values.
    cost: Cost value, a scalar value representing the squared sum of the loss.
    """

    m = x.shape[0]
    
    # Calculate the loss by performing a dot product of input features with parameters and adding the bias term
    loss = np.dot(x, w) + b - y
    
    # Calculate the cost by summing the squared loss values
    cost = np.sum(loss**2) / 2 / m
    
    regularization = lambda_ * np.sum(w**2) / 2 / m

    cost_reg = cost + regularization

    return loss, cost_reg



def CalculateGradient(x, y, w, b, method = "regression", lambda_ = 0):
    """
    Calculates the gradients and cost using the CalculateLoss function.

    Args:
    x: Input features, a 2D numpy array where each row represents a training instance
       and each column represents a feature.
    y: Target values, a 1D numpy array where each element represents the target value
       for a training instance.
    w: Parameter values, a 1D numpy array representing the weights for each feature.
    b: Bias value, a scalar representing the bias term.
    method: Which cost function to use, "regression" or "classification".
    lambda_: Regularization parameter.

    Returns:
    dJ_dw: Gradient of the parameter values with respect to the cost.
    dJ_db: Gradient of the bias value with respect to the cost.
    cost: Cost value.
    """
    m = x.shape[0]  # Number of training instances

    # Calculate the loss and cost using the CalculateLoss function
    if method == "regression":
        loss, cost = CalculateLoss_Regression(x, y, w, b, lambda_ = lambda_)
    else:
        loss, cost = CalculateLoss_Classification(x, y, w, b, lambda_ = lambda_)

    # Calculate the gradients of the parameter values by performing a dot product of the loss and input features
    dJ_dw = np.dot(loss, x) / m + lambda_ * w / m

    # Calculate the gradient of the bias value by summing the loss values
    dJ_db = np.sum(loss) / m

    return dJ_dw, dJ_db, cost



def GD(x, y, iterations = 1000, alpha = 1e-9, method="regression", lambda_ = 0):
    """
    Performs gradient descent optimization to minimize the cost function and update the parameter values.

    Args:
        x: Input features, a 2D numpy array where each row represents a training instance and each column represents a feature.
        y: Target values, a 1D numpy array where each element represents the target value for a training instance.
        iterations: Number of iterations for gradient descent, default is 1000.
        alpha: Learning rate for gradient descent, default is 1e-9.
        method: Cost function to use, either "regression" or "classification", default is "regression".
        lambda_: Regularization parameter, default is 0.

    Returns:
        w: Updated parameter values, a 1D numpy array representing the weights for each feature.
        b: Updated bias value, a scalar representing the bias term.
        costs: List of cost values at each iteration.
    """

    m = x.shape[0]
    n = x.shape[1]

    # Generate initial weights

    w = np.zeros_like(x[0])
    b = 0

    # Initialize log to store steps of the GD
    log = []

    # Perform Gradient Descent
    i = 0
    for i in range(iterations):

        # Calculate gradients
        dJ_dw, dJ_db, cost = CalculateGradient(x, y, w, b, method = method, lambda_ = lambda_)

        # Log the values
        log.append([i, w, b, cost, dJ_dw, dJ_db])

        # adjust the weights and bias
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db

        i += 1

    # Convert list log to numpy
    log = np.array(log, dtype="object")

    return w, b, log