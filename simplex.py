import numpy as np
from numpy import linalg as la

class Problem:
    '''
    Defines an LP problem of the form:\n
    min c.T x\n
    Ax=b\n
    x>=0

    x is initialized at x=0.

    Parameter
    ---------
    A : ndarray(m,n)
        constraint matrix
    b : ndarray(m,)
        constraint vector
    c : ndarray(n,)
        objective vector
    '''
    def __init__(self, A, b, c):
        self.A=A
        self.m,self.n= np.shape(A)
        self.b=b
        self.c=c
        self.min=True
        assert((self.m,self.n)==(np.shape(b)[0],np.shape(c)[0])), "Matrix size did not coincide objective and constraint vector."
        self.x=np.zeros(self.n)
        self.B_idx=np.arange(la.matrix_rank(A))
        self.itr_steps=0

    def __str__(self) -> str:
        if self.min:
            return f"===LP Simplex Solver===\nObjective :\nMin {self.c}\nConstraints :\n--A--\n{self.A}\n--AB--\n{self.A[:,self.B_idx]}\n--b--\n{self.b}\n--x--\n{self.x}\n=======================\n"
        else:
            return f"===LP Simplex Solver===\nObjective :\nMax {-self.c}\nConstraints :\n--A--\n{self.A}\n--AB--\n{self.A[:,self.B_idx]}\n--b--\n{self.b}\n--x--\n{self.x}\n=======================\n"

    def set_max(self):
        '''
        Turn into a maximization problem
        '''
        if self.min:
            self.min=False
            self.c=-self.c

    def set_min(self):
        '''
        Turn into a minimization problem
        '''
        if not self.min:
            self.min=True
            self.c=-self.c

    def transform_problem(self):
        '''
        Transforms constraints in problem such that b>=0 

        Parameter
        ---------
        A : ndarray(m,n)
            constraint matrix of dimension m,n
        b : ndarray(m,)
            constraint vector of dimension m
        '''
        A=self.A
        b=self.b

        assert(np.shape(A)[0]==np.shape(b)[0])
        for i,bi in enumerate(b):
            if bi<0:
                A[i,:]=-A[i,:]
                b[i]=-bi
    
    def set_x(self, x):
        '''
        Set the current solution state of problem

        Parameter
        ---------
        x : ndarray(n, )
            solution vector of dimension n.
        '''
        assert(np.shape(self.x)==np.shape(x))
        self.x = x

    def set_basis(self, basis_idx, var_idx):
        '''
        Set the current basis solution of problem

        Parameter
        ---------
        basis_idx : array-like
            list of indices of the basis to be changed. (1 to m=rank(A))
        val : array-like
            list of indices of the variable to enter basis. (1 to n)
        '''
        assert(len(basis_idx)==len(var_idx))
        for i in basis_idx:
            self.B_idx[i]=var_idx[i]
    
    def reduced_cost_init(self):
        '''
        Initialize the reduced cost (r_cost) of the current state of the problem.
        '''

        self.r_cost=self.c-self.c[self.B_idx]@(np.linalg.inv(self.A[:,self.B_idx])@self.A)

    def objective_value_init(self):
        '''
        Initialize the objective value (obj_val) of the current state of the problem
        '''
        self.obj_val=self.c[self.B_idx]@self.x[self.B_idx]

    def sync(self):
        '''
        Sync the objectives of the current state.
        '''
        self.objective_value_init()
        self.reduced_cost_init()

    def iterate_simplex(self, rule):
        '''
        Perform a single simplex iteration.
        '''

        enter_idx, leaving_idx, leaving_basis_idx = rule(self)

        print(self.r_cost)
        self.r_cost = self.r_cost-(self.r_cost[enter_idx]/self.A[leaving_basis_idx, enter_idx])*self.A[leaving_basis_idx,:]
        print(self.r_cost)
        
        for row in np.arange(np.shape(self.A)[0]):
            if row!= leaving_basis_idx:
                self.A[row,:]=self.A[row,:]-(self.A[row,enter_idx]/self.A[leaving_basis_idx,enter_idx])*self.A[leaving_basis_idx,:]
            else:
                self.A[row,:]=self.A[row,:]/self.A[row,enter_idx]
        
        print(self.A)

        self.B_idx[enter_idx]=True
        self.B_idx[leaving_idx]=False

        print(self.B_idx)



        



def auxillary_problem(p:Problem):
    '''
    Generate auxillary problem

    Parameter
    ---------
    p : Problem
        Problem to which one wants to generate auxillary problem
    '''
    p.transform_problem()
    A=p.A
    b=p.b
    m,n=np.shape(A)

    aux_prob = Problem(
        np.append(A,np.diag(np.ones(m)),axis=1),
        b,
        np.append(np.zeros(n),np.ones(m))
        )

    aux_prob.set_x(np.append(np.zeros(n),b)) #Set solution initially to the following
    aux_prob.set_basis(np.arange(m),np.arange(n,n+m))     

    return aux_prob

def blands_rule(p:Problem):
    '''
    Gives the entering and leaving basis with Bland's rule.
    i.e. minimal index of non-basis that has negative reduced cost enters.
    Minimal index of basis that are candidates to exit.

    Parameter
    ---------
    p : Problem
        LP problem

    Returns
    -------
    (int,int)
        Entering and leaving basis.
    '''

    x_idx=np.arange(np.size(p.x))
    enter_idx=np.min(x_idx[~p.B_idx & (p.r_cost<0)])
    basis_idx=x_idx[p.B_idx]
    leaving_basis_idx=np.argmin(p.x[p.B_idx]/(p.A[:,enter_idx]).flatten())

    leaving_idx=basis_idx[leaving_basis_idx]

    print(enter_idx)
    print(leaving_idx)
    print(leaving_basis_idx)

    return (enter_idx, leaving_idx, leaving_basis_idx)