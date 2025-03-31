import torch
import torch.nn as nn

class AccentReconstructionLoss(nn.Module):
    def __init__(self):
        super(AccentReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, prediction, target_sparcs):
        pred_sparc, sparc_masks = prediction
        pred_sparc, sparc_masks = pred_sparc[:, :12], sparc_masks[:, :12]
        target_sparcs = target_sparcs[:, :12]

        sparc_masks = ~sparc_masks

        pred_sparc = pred_sparc.masked_select(sparc_masks.unsqueeze(-1))
        target_sparcs = target_sparcs.masked_select(sparc_masks.unsqueeze(-1))

        return self.mse_loss(pred_sparc, target_sparcs)

