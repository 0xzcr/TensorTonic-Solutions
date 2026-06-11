import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.asarray(p, dtype='float')
    y = np.asarray(y, dtype='float')

    loss = []
    for i in range(len(p)):
        loss.append(-(1-p[i])**gamma*y[i]*np.log(p[i]) - p[i]**gamma*(1-y[i])*np.log(1-p[i]))

    return np.mean(loss)
    pass