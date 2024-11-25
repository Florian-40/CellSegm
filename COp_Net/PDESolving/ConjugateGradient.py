import numpy as np



# gradient conjugate method to solve linear system in each time step of Crank Nicolson
def conjgrad(A, b, prec, eps, Nmax):
    x = np.zeros(len(b))
    bn = np.linalg.norm(b)
    r = b.copy()
    k = 0
    residu = 1.0; res = [residu]
    while ((residu >= eps) and (k < Nmax)):
        # Application du preconditionneur
        z = prec.apply(r)
        e = np.dot(r, z)
        if k == 0:
            p = z.copy()
        else:
            beta = e/eprev
            #print("beta = ", beta)
            p = beta * p + z

        q = A.dot(p)
        lam = np.dot(p, q)
        alpha = e / lam
        x = x + alpha * p
        r = r - alpha * q
        residu = np.linalg.norm(r) / bn
        k = k+1
        eprev = e
        res.append(residu)

    return x, res



# preconditionneur if necessary (here not useful)
class id_precond:
    def apply(self, x):
        return x
    
    
def solve (delta_t, t_final, u_0, A, B, eps = 1e-8, Nmax = 10**4): 
    t=0
    u_sol = u_0
    size = int(np.sqrt(u_sol.size))
    while t < t_final : 
        t += delta_t
        u_sol = conjgrad(B, A*u_sol, id_precond(), 1e-8, 10000)[0]

    u_sol = u_sol.reshape((size,size))
    return u_sol

    
    
    
    
    
    
    