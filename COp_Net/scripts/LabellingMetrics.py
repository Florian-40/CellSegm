import numpy as np 
import SimpleITK as sitk 
from skimage.segmentation import expand_labels




def MyConnectedComponent(img = sitk.Image) : 
    """
    Obtain cell labels from binary cell mask. 
    img : 2D cell binary mask of shape (1,y-size,x-size).
    """
    dilate = sitk.BinaryDilate(img==0, [3,3,3])
    dilate_label = sitk.ConnectedComponent(1-dilate)
    dilate_label = sitk.GetImageFromArray(expand_labels(sitk.GetArrayFromImage(dilate_label), 10))
    pred_image = sitk.Mask(dilate_label, sitk.Cast(img, dilate_label.GetPixelID()))
    return pred_image


def most_present_value_in(labelled_image : np.array, structure : np.array, idx :int ):
    """
    Remove blood capillary or hemorragic structure in the labelled cell images. 
    labelled_image : labelled cell image of shape (1,y-size,x-size)
    structure : vessel or hemo is available of shape (1, y-size, x-size)
    idx : slice index of the 3D stack 
    """
    structure = structure[idx,:,:]
    all_values = labelled_image[0,(structure == 1)]
    input_values = np.sort(list(set(list(all_values))))
    if len(input_values) ==0 :
        return labelled_image, ([], [])
    if input_values[0]==0: 
        input_values = input_values[1:]
    c = list(map(list(all_values).count, input_values))
    if c != [] : 
        most_present_value = input_values[c.index(max(c))]
        idx = np.where(labelled_image[0,:,:] == most_present_value)
        labelled_image[0,labelled_image[0,:,:] == most_present_value]=0


        return labelled_image, idx
    else : 
        return labelled_image, ([], []) 
    


def evaluate (pred : sitk.Image, image : sitk.Image, target : sitk.Image, error_percentage = 0.05) :
    """
    Evaluate the percentage of correctly labelled cells, the percentage of erroneously merged cells, 
    the percentage of erroneously split cells in the cell masks obtained through nnU-Net only and nnU-Net + COp-Net. 
    Modification of labels on the input and predicted masks to match the labels on the ground truth mask.

    pred : cell instance segmentation obtained through nnU-Net + COp-Net
    image : cell instance segementation obtained through nnU-Net only
    target : ground truth cell instance segmentation
    """
    # Convert sitk image to numpy array
    pred_numpy = sitk.GetArrayFromImage(pred)
    input_numpy = sitk.GetArrayFromImage(image)
    target_numpy = sitk.GetArrayFromImage(target)
    
    max_label = np.max([np.max(pred_numpy), np.max(input_numpy), np.max(target_numpy)]) + 10

    # Modification of labels on the input masks to match the labels on the ground truth mask
    bij_labels_input, input_new, score2_input, score1_input = label_modif(image, input_numpy, target, target_numpy, max_label, error_percentage = error_percentage)

    # Modification of labels on the predicted masks to match the labels on the ground truth mask
    bij_labels_pred, pred_new, score2_pred, score1_pred = label_modif(pred, pred_numpy, target, target_numpy, max_label, error_percentage = error_percentage)
            
    return sitk.GetImageFromArray(input_new), score1_input, score2_input, bij_labels_input, \
sitk.GetImageFromArray(pred_new), score1_pred, score2_pred, bij_labels_pred, 



def label_modif(image : sitk.Image, image_numpy : np.array, target : sitk.Image, target_numpy : np.array, max_label, error_percentage = 0.05) : 
    """
    Modify the labels of the image to match the labels of the target and compute the number of correctly labelled cells, 
    erroneously merged cells and erroneously split cells. 
    For each label in the target image, if there is an overlap over 1-error_percentage between the target label and the image label, 
    then the image label is modified to be equal to the target label. Otherwise, the image label is modified to be over to the max_label value. 

    - image : input or predicted cell instance segmentation 
    - target : ground truth cell instance segmentation
    - image_numpy : numpy version of image ( = sitk.GetArrayFromImage(image))
    - target_numpy : numpy version of target ( = sitk.GetArrayFromImage(target))
    """

    image_new = np.zeros_like(image_numpy)

    bijection_label=[]
    Nb_Of_Split=[]
    Nb_Of_Merged = 0

    # for all label in the target cell instance segmentation
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(target)
    for i in shape_stats.GetLabels() :
        idx = np.where(target_numpy == i)
        check, idx_input, Nb_Of_Merged, Nb_Of_Split = BijectionAndScores(image_numpy, target_numpy, idx, i, Nb_Of_Merged, Nb_Of_Split, error_percentage=error_percentage )
        # if the overlap is over 1-error_percentage, then the label is modified to be equal to the target label
        if check :
            bijection_label.append(i)
            image_new[idx_input] = i
        elif idx_input is not None :
            # in that case, it's an erroneously merged cell
            image_new[idx_input] = max_label + i

    # if the label is not in the target cell instance segmentation, then the label is modified to be over to the max_label value 
    idx = np.where((image_numpy!=0) & (target_numpy==0))
    nb_of_little_structures=0
    for j in list(set(list(image_numpy[(image_numpy!=0) & (target_numpy==0)]))) : 
        if np.sum(target_numpy[image_numpy==j])==0 : 
            nb_of_little_structures+=1
    
    image_new[(image_numpy!=0) & (image_new==0)] = image_numpy[(image_numpy!=0) & (image_new==0)] + 2*max_label
    
    return bijection_label, image_new, Nb_Of_Split, Nb_Of_Merged


def BijectionAndScores(image_numpy, target_numpy, idx, i, Nb_Of_Merged, Nb_Of_Split, error_percentage = 0.05) : 
    """
    Check if the overlap is over 1-error_percentage between the target label and the image label.
    
    - idx : index of the target label
    - i : target label
    """

    all_values = image_numpy[idx]
    input_values = np.sort(list(set(list(all_values))))
    check = False
    idx_input = None

    # the most present value in all_values
    # most_present_value=-10 if the most present value is 0 
    c = list(map(list(all_values).count, input_values))
    most_present_value = input_values[c.index(max(c))]
    if most_present_value == 0 and len(input_values) ==1 :
        most_present_value=-10
        input_values = []
        
    if most_present_value == 0 and len(input_values) !=1:
        Nb_Of_Merged +=1 
        input_values = [0]
        

    # If there is an overlap with a cell label of image_numpy which is over 1-error_precentage  
    if len(list(all_values[all_values!=most_present_value])) <= error_percentage*len(list(idx[0])) and most_present_value !=0:
        input_values = [most_present_value]
        idx_input = np.where(image_numpy == input_values[0])


        ### check the number of labels in idx_input on target image
        all_values = target_numpy[idx_input]
        target_values = np.sort(list(set(list(all_values))))

        # the most present value in all_values 
        # most_present_value = -10 if all_values contains only 0 
        c = list(map(list(all_values).count, target_values))
        most_present_value = target_values[c.index(max(c))]
        if most_present_value == 0 : 
            most_present_value=-10

        # If the overlap with the corresponding cell label of target_numpy is over 1-error_percentage
        if len(list(all_values[all_values!=most_present_value])) <= error_percentage*len(list(idx[0])) and most_present_value == i:
            target_values = [most_present_value]
            check = True 
        else : 
            # in that case, it's an erroneously merged cell
            Nb_Of_Merged +=1  

    # if input_values contain more than 1 value, it's an erroneously split cell
    Nb_Of_Split.append(len(input_values)) 

    return check, idx_input, Nb_Of_Merged, Nb_Of_Split