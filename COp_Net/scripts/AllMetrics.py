import numpy as np 
import SimpleITK as sitk 
import matplotlib.pyplot as plt

from COp_Net.scripts.LabellingMetrics import MyConnectedComponent, most_present_value_in, evaluate
from COp_Net.scripts.SegmentationMetrics import NSD, clDice_over_slice
from COp_Net.scripts import Visualisation
from COp_Net.scripts.Symetry import symmetry_on_border


class Results:
    def __init__(self, pred : np.array, labels : np.array, image : np.array, background_orig : str = None, hemo : np.array = None, vessel : np.array = None) :
        """
        Inputs are numpy arrays of shape (z-size, y-size, x-size) or (y-size, x-size).

        Args: 
        - pred : binary cell contour segmentation or cell contour probability map predicted by the COp-Net
        - labels : ground truth cell contour segmentation
        - image : binary cell contour segmentation or cell contour probability map predicted by nnU-Net only (Step #1)
        - background_orig : path to the original grayscale image with black borders, if symetry is necessary
        - hemo : binary mask of the hemorragic zone, if available
        - vessel : binary mask of the blood capillary, if available
        
        """
        

        # Post processing
        pred = np.round(pred)
        image = np.round(image)
        pred[image == 1]=1
    
        image[image == 0] =0
        image[image == 1] =1

        x_size = image.shape[-1]
        y_size = image.shape[-2]
        
        # Reshape in case of 2D input images
        if len(image.shape) == 2 :
            pred_image = pred.reshape(1, y_size, x_size)
            init_image = image.reshape(1, y_size, x_size)
            final_image = labels.reshape(1, y_size, x_size)
            if vessel is not None :
                vessel = vessel.reshape(1, y_size, x_size)
            if hemo is not None :
                hemo = hemo.reshape(1, y_size, x_size)
        else : 
            pred_image = pred
            init_image = image
            final_image = labels

        # Symmetry on border if necessary (teh original grayscale image contains black borders due to the alignement process)
        if background_orig is not None :
            background = sitk.ReadImage(background_orig)
            background = sitk.GetArrayFromImage(background)
            if len(background.shape) == 2 :
                background = background.reshape(1, background.shape[0], background.shape[1])
            init_image, pred_image = symmetry_on_border(background, [init_image, pred_image])[1]


        # Convert to np.uint16
        pred_image = np.uint16(pred_image)
        init_image = np.uint16(init_image)
        final_image = np.uint16(final_image)

        # Apply a morphological closing operation to clean noisy cell predictions
        pred_image = sitk.GetImageFromArray(pred_image)
        final_image = sitk.GetImageFromArray(final_image)
        for idx in range(0, len(init_image)) : 
            pred_image[:,:,idx] = sitk.BinaryClosingByReconstruction(pred_image[:,:,idx], [3,3,3])
            final_image[:,:,idx] = sitk.BinaryClosingByReconstruction(final_image[:,:,idx], [3,3,3])
        pred_image = sitk.GetArrayFromImage(pred_image)
        final_image = sitk.GetArrayFromImage(final_image)

        self.pred_image = pred_image
        self.init_image = init_image
        self.final_image = final_image
        self.hemo = hemo
        self.vessel = vessel
        self.x_size = x_size
        self.y_size = y_size


    def LabellingMetrics(self, error_percentage = 0.15, slice_visualization = 0) :
        """
        Evaluate the performance of the COp-Net and nnU-Net only with the percentage of correctly labelled cells, erroneously merged cells 
        and erroneously split cells.

        Args: 
        - error_percentage : percentage of the overlap error allowed between two corresponding cell labels. 
        """

        # Correctly labelled cells
        bijection_pred=[]
        bijection_init=[]
        # Erroneously merged cells
        merged_pred_list = []
        merged_init_list = []
        # Erroneously split cells
        split_pred_list = []
        split_init_list = []
        
        # Number of cells in the ground truth cell instance segmentation
        total_nb_of_cells = 0
        
        for idx in range(0, len(self.init_image)) :
            pred_image = sitk.GetImageFromArray(np.uint16(self.pred_image[idx,:,:]).reshape(1, self.y_size, self.x_size))
            init_image = sitk.GetImageFromArray(np.uint16(self.init_image[idx,:,:]).reshape(1, self.y_size, self.x_size))
            final_image = sitk.GetImageFromArray(np.uint16(self.final_image[idx,:,:]).reshape(1, self.y_size, self.x_size))

            if slice_visualization == idx : 
                # Saving results 
                sitk.WriteImage(sitk.GetImageFromArray(sitk.GetArrayFromImage(pred_image)[0,:,:]), 'Outputs/COpNet_contour.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(sitk.GetArrayFromImage(init_image)[0,:,:]), 'Outputs/nnUNet_contour.nii.gz')  

            # Get cell masks
            pred_image = sitk.ChangeLabel(pred_image, {1:0, 0:1})
            init_image = sitk.ChangeLabel(init_image, {1:0, 0:1})
            final_image = sitk.ChangeLabel(final_image, {1:0, 0:1})

            # Get cell mask contours
            if slice_visualization == idx :
                self.pred_contour = sitk.BinaryContour(pred_image)
                self.init_contour = sitk.BinaryContour(init_image)
                self.final_contour = sitk.BinaryContour(final_image)  

            
            # Apply the connected component algorithm
            pred_image = MyConnectedComponent(pred_image)
            init_image = MyConnectedComponent(init_image)
            final_image = sitk.ConnectedComponent(final_image)
            
            # Convert the images to numpy array
            pred_image = sitk.GetArrayFromImage(pred_image)
            init_image = sitk.GetArrayFromImage(init_image)
            final_image = sitk.GetArrayFromImage(final_image)
            
            # remove the label in the blood capillary and hemorragic zone
            if self.vessel is not None :
                #final_image, idx_vessel_final = most_present_value_in(final_image, self.vessel, idx)
                #pred_image[0, idx_vessel_final[0], idx_vessel_final[1]] = 0
                #init_image[0, idx_vessel_final[0], idx_vessel_final[1]] = 0
                final_image[0,self.vessel[idx,:,:]==1]=0
                pred_image[0,self.vessel[idx,:,:]==1]=0
                init_image[0,self.vessel[idx,:,:]==1]=0
            if self.hemo is not None : 
                final_image, self.idx_hemo_final = most_present_value_in(final_image, self.hemo, idx)
                pred_image[0, self.idx_hemo_final[0], self.idx_hemo_final[1]] = 0
                init_image[0, self.idx_hemo_final[0], self.idx_hemo_final[1]] = 0

            pred_image = sitk.GetImageFromArray(pred_image)
            init_image = sitk.GetImageFromArray(init_image)
            final_image = sitk.GetImageFromArray(final_image)
            
            #### Compute the number of cells on the ground truth image
            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(final_image)
            target_number_labels = shape_stats.GetNumberOfLabels()
            
            # compute the labelling metrics on the current image (proportion of correctly labelled cells, 
            # proportion of erroneously merged cells, proportion of erroneously split cells)
            self.init_image_eval, merged_input, split_input, bij_labels_input, self.pred_image_eval,\
    merged_pred, split_pred, bij_labels_pred = evaluate(pred_image, init_image, final_image, error_percentage = error_percentage)
            
            
            if slice_visualization == idx : 
                self.bij_labels_pred = bij_labels_pred
                self.bij_labels_input = bij_labels_input
                self.final_image_inst = final_image
                Results.Visualization(self, idx=idx)




            total_nb_of_cells += target_number_labels
            
            bijection_pred.append(len(bij_labels_pred)/target_number_labels)
            bijection_init.append(len(bij_labels_input)/target_number_labels)
            
            merged_pred_list.append(merged_pred/target_number_labels)
            merged_init_list.append(merged_input/target_number_labels)
            
            split_pred_list.append(np.sum(np.array(split_pred)!=1)/target_number_labels)
            split_init_list.append(np.sum(np.array(split_input)!=1)/target_number_labels)
                
        result =  np.mean(bijection_pred), np.mean(bijection_init), np.mean(merged_pred_list), np.mean(merged_init_list), \
        np.mean(split_pred_list), np.mean(split_init_list), np.std(bijection_pred), \
        np.std(bijection_init), np.std(merged_pred_list), np.std(merged_init_list), np.std(split_pred_list), np.std(split_init_list)
        
        # print
        if len(self.init_image) > 1 :
            print('Total number of cells: ', total_nb_of_cells)
            print('---------------  Validation on {} images -------------- \n'.format(len(init_image)))
            print('nnUNet + COp-Net: ')
            print('Proportion of correctly labelled cells : {} +/- {}'.format(np.round(result[0],3), np.round(result[6],3)))
            print('-------------------------------------------------------------------')
            print('Proportion of erroneously merged component: {} +/- {}'.format(np.round(result[2],3), np.round(result[8],3)))
            print('Proportion of erroneously split component : {} +/- {}'.format(np.round(result[4],3), np.round(result[10],3)))
            print('-------------------------------------------------------------------\n')

            print('nnUNet only: ')
            print('Proportion of correctly labelled cells : {} +/- {}'.format(np.round(result[1],3), np.round(result[7],3)))
            print('-------------------------------------------------------------------')
            print('Proportion of erroneously merged component : {} +/- {}'.format(np.round(result[3],3), np.round(result[9],3)))
            print('Proportion of erroneously split component : {} +/- {}'.format(np.round(result[5],3), np.round(result[11],3)), '\n')
            print('------------------------------------------------------------------- \n')
        else : 
            print('Total number of cells: ', total_nb_of_cells)
            print('---------------  Validation on 1 image -------------- \n')
            print('nnUNet + COp-Net: ')
            print('Proportion of correctly labelled cells : {}'.format(np.round(result[0],3)))
            print('-------------------------------------------------------------------')
            print('Proportion of erroneously merged component: {}'.format(np.round(result[2],3)))
            print('Proportion of erroneously split component : {}'.format(np.round(result[4],3)))
            print('-------------------------------------------------------------------\n')

            print('nnUNet only: ')
            print('Proportion of correctly labelled cells : {}'.format(np.round(result[1],3)))
            print('-------------------------------------------------------------------')
            print('Proportion of erroneously merged component : {}'.format(np.round(result[3],3)))
            print('Proportion of erroneously split component : {}'.format(np.round(result[5],3)), '\n')
            print('------------------------------------------------------------------- \n')
        
        self.labelling_metrics = result
        return result
        

    def SegmentationMetrics(self, NSD_threshold = [2]): 
        """
        Compute the Normalized Surface Dice (NSD) and the Centerline Dice (CLDice) metrics.
        
        """

        # NSD
        NSD_pred = NSD(self.pred_image, self.final_image, NSD_threshold)
        NSD_init = NSD(self.init_image, self.final_image, NSD_threshold)

        if len(self.init_image)>1 :
            print('Normalized Surface Dice (NSD): \n')
            print('nnUNet + COp-Net: ', NSD_pred.mean(), ' +/- ', NSD_pred.std()) 
            print('nnUNet only: ', NSD_init.mean(), ' +/- ', NSD_init.std(), '\n')
        else : 
            print('Normalized Surface Dice (NSD): \n')
            print('nnUNet + COp-Net: ', NSD_pred.item()) 
            print('nnUNet only: ', NSD_init.item(), '\n')

        # CLDice
        CLDice_pred = clDice_over_slice(self.pred_image, self.final_image)
        CLDice_init = clDice_over_slice(self.init_image, self.final_image)

        if len(self.init_image)>1 :
            print('Centerline Dice (clDice): \n')
            print('nnUNet + COp-Net: ', np.mean(CLDice_pred), ' +/- ', np.std(CLDice_pred))
            print('nnUNet only: ', np.mean(CLDice_init), ' +/- ', np.std(CLDice_init), '\n')
        else : 
            print('Centerline Dice (clDice): \n')
            print('nnUNet + COp-Net: ', np.mean(CLDice_pred))
            print('nnUNet only: ', np.mean(CLDice_init), '\n')


    def Visualization(self, idx : int = 0): 
        
        plt.figure(figsize=(8,5))
        plt.subplot(1,3,1)
        plt.imshow(self.init_image[idx,:,:])
        plt.title('nnU-Net')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(self.pred_image[idx,:,:])
        plt.title('nnU-Net + COp-Net')
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.imshow(self.final_image[idx,:,:], cmap='gray')
        plt.title('Ground truth')
        plt.axis('off')
        plt.show()



        ## Get list of labels for each image 
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        # prediction
        shape_stats.Execute(self.pred_image_eval)
        pred_labels = list(shape_stats.GetLabels())
        # initial
        shape_stats.Execute(self.init_image_eval)
        init_labels = list(shape_stats.GetLabels())


        ## Select correctly labelled cells and erroneosuly labelled cells
        relabelMap = {int(i) : 0 for i in pred_labels if i in self.bij_labels_pred}
        pred_non_bij_image_eval = sitk.ChangeLabel(self.pred_image_eval, changeMap = relabelMap)
        relabelMap = {int(i) : 0 for i in pred_labels if i not in self.bij_labels_pred}
        self.pred_image_eval = sitk.ChangeLabel(self.pred_image_eval, changeMap = relabelMap)
        
        relabelMap = {int(i) : 0 for i in init_labels if i in self.bij_labels_input}
        init_non_bij_image_eval = sitk.ChangeLabel(self.init_image_eval, changeMap = relabelMap)
        relabelMap = {int(i) : 0 for i in init_labels if i not in self.bij_labels_input}
        self.init_image_eval = sitk.ChangeLabel(self.init_image_eval, changeMap = relabelMap)

        # background image for the visualization
        background = sitk.GetImageFromArray(np.uint8(220*np.ones((1,self.y_size,self.x_size))))

        # Add contour on background for init, pred and final 
        background_init_contour = sitk.GetArrayFromImage(background)
        background_pred_contour = background_init_contour.copy()
        background_final_contour = background_init_contour.copy()
        
        pred_contour = sitk.GetArrayFromImage(self.pred_contour)
        init_contour = sitk.GetArrayFromImage(self.init_contour)
        final_contour = sitk.GetArrayFromImage(self.final_contour)
        
        pred_non_bij_image_eval = sitk.GetArrayFromImage(pred_non_bij_image_eval)
        init_non_bij_image_eval = sitk.GetArrayFromImage(init_non_bij_image_eval)
        
        background_init_contour[init_contour==1] = 0
        background_init_contour[init_non_bij_image_eval>0] = 0
        background_pred_contour[pred_contour==1] = 0
        background_pred_contour[pred_non_bij_image_eval>0] = 0
        background_final_contour[final_contour==1] = 0

        if self.vessel is not None : 
            #background_init_contour[0,idx_vessel_final[0], idx_vessel_final[1]] = 100
            #background_pred_contour[0,idx_vessel_final[0], idx_vessel_final[1]] = 100
            #background_final_contour[0,idx_vessel_final[0], idx_vessel_final[1]] = 100
            background_init_contour[0,self.vessel[idx,:,:]==1] =100
            background_pred_contour[0,self.vessel[idx,:,:]==1] =100
            background_final_contour[0,self.vessel[idx,:,:]==1] =100
        if self.hemo is not None : 
            background_init_contour[0,self.idx_hemo_final[0],self.idx_hemo_final[1]]  =100
            background_pred_contour[0,self.idx_hemo_final[0],self.idx_hemo_final[1]] = 100
            background_final_contour[0,self.idx_hemo_final[0],self.idx_hemo_final[1]] = 100


        background_init_contour = sitk.GetImageFromArray(background_init_contour)
        background_pred_contour = sitk.GetImageFromArray(background_pred_contour)
        background_final_contour = sitk.GetImageFromArray(background_final_contour)


        # afficher le résultat
        Visualisation.MultiImageDisplay(image_list=[sitk.LabelOverlay(background_init_contour, self.init_image_eval), 
                                  sitk.LabelOverlay(background_pred_contour, self.pred_image_eval),
                                  sitk.LabelOverlay(background_final_contour,self.final_image_inst) ], 
                      title_list=["nnU-Net", "nnU-Net + COp-Net", "Ground Truth"], 
                      figure_size=(9, 6), 
                     horizontal=True, 
                     shared_slider=True); 




    

    

    



