

def eval_metrics(target, pred_prob):
    """
        `pred_prob` must be numpy.ndarray or torch.Tensor.
        `pred_prob` should be the probabilities, instead of binary classification results.
        If `pred_prob` is not probability, then the `rocauc` would be equivalent to classification accuracy? (No)
        """