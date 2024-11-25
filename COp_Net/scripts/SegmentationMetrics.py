import numpy as np 
import torch
from monai.metrics import compute_surface_dice
from skimage.morphology import skeletonize






def NSD(image : np.array, label : np.array , threshold = [2]) :
    """
    image and label are numpy 2D or 3D arrays of shape (y_size, x_size) or (z_size, y_size, x_size), respectively. 
    They represent the predicted and ground truth binary masks of the cell contour segmentation.
    
    """
    # if the input is 2D, add a third dimension
    if len(image.shape) == 2:
        image = image.resize(1, image.shape[0], image.shape[1])
        label = label.resize(1, label.shape[0], label.shape[1])

    # The input in the compute_surface_dice function must be a pytorch tensor
    image_inv = torch.Tensor(1-image)
    label_inv = torch.Tensor(1-label)

    # one hot encoding 
    image1_one_hot = torch.zeros((image_inv.shape[0], 2, image_inv.shape[-2], image_inv.shape[-1]))
    image1_one_hot[:,0,:,:] = image_inv
    image1_one_hot[:,1,:,:] = torch.Tensor(image)

    label_one_hot = torch.zeros((label_inv.shape[0], 2, label_inv.shape[-2], label_inv.shape[-1]))
    label_one_hot[:,0,:,:] = label_inv
    label_one_hot[:,1,:,:] = torch.Tensor(label)

    # Compute the Normalized Surface Dice (NSD)
    NSD_output = compute_surface_dice(image1_one_hot, label_one_hot, threshold)
    
    return NSD_output


# the 3 functions usefull to compute the centerline dice metric
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    return 2*tprec*tsens/(tprec+tsens)


def clDice_over_slice(v_p,v_l):
    cldice=[]
    for i in range(0, len(v_p)):
        cldice.append(clDice(v_p[i,:,:], v_l[i,:,:]))
    return cldice