import SimpleITK as sitk 
import numpy as np


def get_background_border_for_one_slice(background_numpy : np.array, slice : int) : 
    """
    This function return [xmin, xmax, ymin, ymax] of only ith image, without black border. 
    Warning, xmax and ymax is the last line and column image before the border. 
    - background_numpy : 3d np.array of background image.
    """

    i= slice

    x1=0
    y1=0
    x2 = background_numpy.shape[2]-1
    y2 = background_numpy.shape[1]-1
    while background_numpy[i,:, x1].any()==0 : 
        x1 = x1+1
    while background_numpy[i,y1, :].any()==0 : 
        y1 = y1+1
    while background_numpy[i,:,x2].any() ==0 : 
        x2=x2-1
    while background_numpy[i,y2,:].any()==0 : 
        y2=y2-1

    return [x1, x2, y1, y2]



def symmetry_on_border (background_numpy : np.array, mask_list : list = None):
    """
    Put symmetry on black border of image.
    """

    for i in range(len(background_numpy)) : 
        xmin, xmax, ymin, ymax = get_background_border_for_one_slice(background_numpy, int(i))
        background_numpy[i,:, 0:xmin] = background_numpy[i,:,2*xmin:xmin:-1]
        background_numpy[i,:,xmax+1:] = background_numpy[i,:,xmax-1:2*xmax-background_numpy.shape[2]:-1]

        background_numpy[i,0:ymin,:] = background_numpy[i,2*ymin:ymin:-1,:]
        background_numpy[i,ymax+1:, :] = background_numpy[i,ymax-1:2*ymax-background_numpy.shape[1]:-1,:]

        if mask_list is not None : 
            k=0
            for mask in mask_list : 
                mask_numpy = mask
                mask_numpy[i,:, 0:xmin] = mask_numpy[i,:,2*xmin:xmin:-1]
                mask_numpy[i,:,xmax+1:] = mask_numpy[i,:,xmax-1:2*xmax-mask_numpy.shape[2]:-1]

                mask_numpy[i,0:ymin,:] = mask_numpy[i,2*ymin:ymin:-1,:]
                mask_numpy[i,ymax+1:, :] = mask_numpy[i,ymax-1:2*ymax-mask_numpy.shape[1]:-1,:]
                mask_list[k] = mask_numpy
                k += 1
    
    
    return background_numpy, mask_list