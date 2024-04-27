
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        intersection = torch.logical_and(y_true, y_pred)
        union = torch.logical_or(y_true, y_pred)
        loss = 1 - (2 * torch.sum(intersection, dim=0)) / (torch.sum(union, dim=0) + 1e-7)
        loss = torch.mean(loss)
        return loss


class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()

    def forward(self, cosine_sim_matrix):
        logits = cosine_sim_matrix
        exp_logits = torch.exp(logits)        
        diag_logits = torch.diag(exp_logits)
        #get the sum of the exponential of the logits
        exp_logits_sum = exp_logits.sum(1)
        #compute the loss
        loss = -torch.log(diag_logits / exp_logits_sum)
        #compute the mean loss
        loss = loss.mean()
        return loss