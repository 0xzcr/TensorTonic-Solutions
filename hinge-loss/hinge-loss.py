import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.asarray(y_true, dtype ='float')
    y_score = np.asarray(y_score, dtype ='float')
    losses = []
    for i in range(len(y_true)):
        losses.append(np.maximum(0, (margin-y_true[i]*y_score[i])))
    if reduction =='mean':
        return np.mean(losses)
    else:
        return np.sum(losses)
    pass