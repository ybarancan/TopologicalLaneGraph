import torch
import torch.nn.functional as F
import numpy as np
INV_LOG2 = 0.693147


# def memory_distance(q,m1,m2):
#     '''
#     m1 and m2 are tuples of key-value matrices
#     '''
    
#     keys1, values1 = m1
#     keys2, values2 = m2
    
    


def balanced_binary_cross_entropy(logits, labels, mask, weights):
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)


def uncertainty_loss(x, mask):
    """
    Loss which maximizes the uncertainty in invalid regions of the image
    """
    labels = ~mask
    x = x[labels.unsqueeze(1).expand_as(x)]
    xp, xm = x, -x
    entropy = xp.sigmoid() * F.logsigmoid(xp) + xm.sigmoid() * F.logsigmoid(xm)
    return 1. + entropy.mean() / INV_LOG2


def prior_uncertainty_loss(x, mask, priors):
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean() 


def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def focal_loss(logits, labels, mask, alpha=0.5, gamma=2):
    
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), 
                                                  reduce=False)
    pt = torch.exp(-bce_loss)
    at = pt.new([alpha, 1 - alpha])[labels.long()]
    focal_loss = at * (1 - pt) ** gamma * bce_loss

    return (focal_loss * mask.unsqueeze(1).float()).mean()


def soft_focal_loss(logits, labels, mask, alpha=0.5, gamma=2):
    
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), 
                                                  reduce=False)
    pt = torch.exp(-bce_loss)
    at = pt.new([alpha, 1 - alpha])[labels.long()]
    focal_loss = at * (1 - pt) ** gamma * bce_loss

    return (focal_loss * mask.unsqueeze(1).float()).mean()


def prior_offset_loss(logits, labels, mask, priors):

    priors = logits.new(priors).view(-1, 1, 1)
    prior_logits = torch.log(priors / (1 - priors))
    labels = labels.float()

    weights = .5 / priors * labels + .5 / (1 - priors) * (1 - labels)
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits - prior_logits, labels, 
                                              weights)

def iou_calculator(annotation, segmentation, void_pixels=None):
    """
    annotation : gt mask
    segmentation : method estimate
    """
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def miou(estimates, gt, one_hot=False, n_objects=None):
    '''
    The inputs SHOULD NOT include background
    '''
    if one_hot:
        assert ((len(estimates.shape) == 4) & (len(gt.shape) == 4))
        
        ious=[]
        
        for t in range(gt.shape[0]):
            temp_ious = []
            for k in range(gt.shape[1]):
                cur_est = estimates[t,k,...]
                cur_gt = gt[t,k,...]
                                
                temp_ious.append(iou_calculator(cur_gt,cur_est))
            ious.append(np.array(temp_ious))
        return ious
    
    else:
#        assert ((len(estimates.shape) == 3) & (len(gt.shape) == 3))
        
        if not n_objects:
            n_objects = np.max(gt)    
        
        ious=[]
        
        for t in range(gt.shape[0]):
            temp_ious = []
            for k in range(1,n_objects+1):
                cur_est = estimates[t,...] == k
                cur_gt = gt[t,...] == k
                                
                temp_ious.append(iou_calculator(cur_gt,cur_est))
            ious.append(np.array(temp_ious))
        return ious
    

