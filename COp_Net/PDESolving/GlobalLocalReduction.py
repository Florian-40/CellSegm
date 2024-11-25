import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt



def norme(radius : int) : 
    """
    Compute the local distance filter
    """
    x = np.linspace(-radius, radius, radius*2)
    y = np.linspace(-radius, radius, radius*2)
    filter_norme = np.zeros((radius*2, radius*2))
    for k in range(0, filter_norme.shape[0]) :
        for l in range(0, filter_norme.shape[1]) : 
            filter_norme[k,l] = np.linalg.norm([x[k],y[l]])/radius

    filter_norme = 1-filter_norme
    filter_norme[filter_norme<=0] = 0
    return filter_norme


def norme_adapted_size(idx_center, radius, size ) :
    filter_norme = norme(radius)
    
    # adapt the size of the norme filter to correspond to the size of the real square (in case of square on the border)
    y_max = 2*radius - ((idx_center[0]+radius - size)>0)*(idx_center[0]+radius - size)
    y_min = ((idx_center[0]-radius)<0)*(radius - idx_center[0])

    x_max = 2*radius - ((idx_center[1]+radius - size)>0)*(idx_center[1]+radius - size)
    x_min = ((idx_center[1]-radius)<0)*(radius - idx_center[1])

    filter_norme = filter_norme[y_min:y_max, x_min:x_max]
    return filter_norme
    
    

class GlobalLocalReduction () : 
    def __init__(self, ground_truth_image : np.array, alpharange : list, betamax : float, N1 : int, N2 : int, radius : list) :
        """
        ground_truth_image : must be a squarred 2D-image 
        alpharange : list of two elements [alphamin, alphamax] with alphamin < alphamax.
        radius : list of two elements with alphamin < alphamax.
        N1 : number of local reduction of cell contour probability (diffusion)
        N2 : number of local dropout of cell contour probability (signal loss)
        """
        self.u_0 = ground_truth_image.copy()
        
        # use to remove area with local disruption and avoid multiple diruption in the same area
        self.GT_image = ground_truth_image.copy() 
        
        self.size = ground_truth_image.shape[0]
        self.alphamin = alpharange[0]
        self.alphamax = alpharange[1]
        self.betamax = betamax
        self.N1 = N1
        self.N2 = N2
        
        self.random_radius = stats.randint.rvs(radius[0], radius[1], size = self.N1+self.N2)
    
    

        
        
    def add_alpha_area(self,i) :
        """

        """
        
        # select radonmly a center in GT_image cell contour
        idx = np.where(self.GT_image >= 0.5)
        random_idx = int(random.choice(np.arange(0, len(idx[0]))))
        idx_center = [idx[0][random_idx], idx[1][random_idx]]
        
        radius = self.random_radius[i]
        norme_alpha = norme_adapted_size(idx_center, radius, self.size)
        
        self.alpha[np.max([0,idx_center[0]-radius]) : idx_center[0]+radius , 
          np.max([0,idx_center[1]-radius]) : idx_center[1]+radius] += (self.alphamax-self.alphamin)*norme_alpha
        
        self.GT_image[np.max([0,idx_center[0]-radius]) : idx_center[0]+radius , 
              np.max([0,idx_center[1]-radius]) : idx_center[1]+radius]=0
        
        
    def add_beta_area(self,i) :
        """

        """
        
        # select radonmly a center in GT_image cell contour
        idx = np.where(self.GT_image >= 0.5)
        random_idx = int(random.choice(np.arange(0, len(idx[0]))))
        idx_center = [idx[0][random_idx], idx[1][random_idx]]
        
        radius = self.random_radius[i]
        norme_beta = norme_adapted_size(idx_center, radius, self.size)
        
        self.beta[np.max([0,idx_center[0]-radius]) : idx_center[0]+radius , 
          np.max([0,idx_center[1]-radius]) : idx_center[1]+radius] += norme_beta*self.betamax
        
        self.GT_image[np.max([0,idx_center[0]-radius]) : idx_center[0]+radius , 
              np.max([0,idx_center[1]-radius]) : idx_center[1]+radius]=0
        
        
    
    
    def AllAreaReduction(self, display_area = True) :
        i = 0
        # alpha and beta initialization
        self.alpha = self.alphamin*np.ones((self.size,self.size))
        self.beta = np.zeros((self.size, self.size))
        
        while i<self.N1 :
            GlobalLocalReduction.add_alpha_area(self,i)
            i += 1
            
        while i<self.N1+self.N2 : 
            GlobalLocalReduction.add_beta_area(self,i)
            i += 1
            
            
        if display_area :  
            plt.figure(figsize=(9,3))

            plt.subplot(131)
            plt.imshow(self.u_0, cmap='gray')
            plt.title('Initial image')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(self.alpha)
            plt.title('Alpha term')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(self.beta)
            plt.title('Beta term')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.show()
            
            

    def Vectorization(self) : 
        """
        Vectorization of alpha and beta term and u_0 with Neumann boundary condition.
        """
        
        u_sol = np.zeros((self.size + 2, self.size + 2))
        alpha_border = np.zeros((self.size + 2, self.size + 2))
        beta_border = np.zeros((self.size + 2, self.size + 2))

        u_sol[1 : self.size+1 , 1: self.size+1 ] = self.u_0
        alpha_border[1 : self.size+1 , 1: self.size+1] = self.alpha
        beta_border[1 : self.size+1 , 1: self.size+1] = self.beta
        
        self.u_0 = u_sol.reshape(u_sol.size)
        self.alpha = alpha_border.reshape(alpha_border.size)
        self.beta = beta_border.reshape(beta_border.size)
        
        

        






    



    
    
    
    
    
    