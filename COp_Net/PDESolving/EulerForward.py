import numpy as np 
import scipy.sparse as sp
import matplotlib.pyplot as plt


class EulerForward(): 
    def __init__(self, u_0, alpha, beta, t_final : float, delta_t : float, delta_x=1 ) :
        """
        Forward Euler time scheme with finite differences. 
        
        - u_0 : ground truth vectorized image with Neumann boundary condition (GlobalLocalReduction)
        - alpha : vectorized alpha term (GlobalLocalReduction)
        - beta : vectorized beta term (GlobalLocalReduction)
        """
        self.u_0 = u_0.copy()
        self.size = u_0.size
        
        self.t_final = t_final 
        self.delta_t = delta_t
        self.delta_x = delta_x
        
        
        # Crank Nicolson time scheme matrix construction
        # U^{n+1} = A U^{n}
        sqrt_size = int(np.sqrt(self.size))
        
        # A sparse matrix construction 
        data = np.zeros([5, self.size])
        data[0,:] = (1-self.delta_t/(self.delta_x**2)*(4*alpha+beta))
        data[1,0:-1] = self.delta_t/(self.delta_x**2)*alpha[1:]
        data[2,1:] = self.delta_t/(self.delta_x**2)*alpha[:-1]
        data[3,0:-sqrt_size ] = self.delta_t/(self.delta_x**2)*alpha[sqrt_size:]
        data[4,sqrt_size:] = self.delta_t/(self.delta_x**2)*alpha[:-sqrt_size]
        A = sp.spdiags(data, [0, -1, 1, -sqrt_size, sqrt_size], self.size, self.size)
        self.A = A.tocsr()
        
        
    def Solve(self, display_sol = True, eps = 1e-8, Nmax = 10**4) :
        t=0
        u_sol = self.u_0
        size = int(np.sqrt(u_sol.size))
        while t < self.t_final : 
            t += self.delta_t
            u_sol = self.A*u_sol
            
        u_sol = u_sol.reshape((size,size))
        self.u_sol = u_sol[1 : u_sol.shape[0]-1 , 1: u_sol.shape[1]-1 ]
                
        sqrt_size = int(np.sqrt(self.size))
        u_0 = self.u_0.reshape(sqrt_size, sqrt_size)
        u_0 = u_0[1 : u_0.shape[0]-1 , 1: u_0.shape[1]-1 ]
        
        if display_sol : 
            plt.figure()

            plt.subplot(121)
            plt.imshow(u_0, cmap='gray')
            plt.title('Initial image')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(self.u_sol, vmin=0, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title('Simulated probability map')
            plt.show()
            #print('minimal value = ', np.min(self.u_sol))
            #print('maximal value = ', np.max(self.u_sol))
        
        return self.u_sol
        
        
        
        
        
        
        
        
        
        