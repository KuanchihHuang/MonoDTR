import torch
import torch.nn.functional as F


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = 0

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices

class DepthFocalLoss(object):
  
    def __init__(self, max_depth=192, start_depth=0, focal_coefficient=0.0):
        self.max_depth = max_depth
        self.start_depth = start_depth
        self.end_depth = start_depth + max_depth - 1
        self.focal_coefficient = focal_coefficient
        self.eps = 1e-40
        self.variance = 0.5

    def __call__(self, estCost, gtDepth):
        
        N, C, H, W = estCost.shape
        scaled_gtDepth = gtDepth.clone() #N, 1, H, W
       
        lower_bound = self.start_depth
        upper_bound = lower_bound + int(self.max_depth)
        mask = (scaled_gtDepth > lower_bound) & (scaled_gtDepth < upper_bound)
        mask = mask.detach_().type_as(scaled_gtDepth)
        
        if mask.sum() < 1.0:
            scaled_gtProb = torch.zeros_like(estCost)  # let this sample have loss with 0
        else:
            gtDepth = scaled_gtDepth * mask
            
            index = torch.linspace(self.start_depth, self.end_depth, self.max_depth)
            index = index.to(gtDepth.device)
            index = index.repeat(N, H, W, 1).permute(0, 3, 1, 2).contiguous()

            mask = (gtDepth > self.start_depth) & (gtDepth < self.end_depth)
            mask = mask.detach().type_as(gtDepth)
            gtDepth = gtDepth * mask
        
            scaled_distance = ((-torch.abs(index - gtDepth)) / self.variance)
            probability = F.softmax(scaled_distance, dim=1)
            scaled_gtProb = probability * mask + self.eps

        estProb = F.log_softmax(estCost, dim=1)
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estProb) * weight * mask.float()).sum(dim=1, keepdim=True).mean()
        
        return loss
