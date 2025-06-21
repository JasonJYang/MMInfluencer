import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def cross_entropy_loss(output, target):
    """
    Cross entropy loss for multi-class classification.
    
    Args:
        output: logits of shape [batch_size, num_classes]
        target: class labels of shape [batch_size]
    """
    return F.cross_entropy(output, target)
