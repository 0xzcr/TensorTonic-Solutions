import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred, dtype='float')
    y_true = np.asarray(y_true, dtype='float')

    subs= []
    for i in range(len(y_pred)):
        subs.append((y_pred[i]-y_true[i])**2)
    return np.mean(subs)
    pass
    
