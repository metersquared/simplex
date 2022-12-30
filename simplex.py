import numpy as np

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
        self.b=b
        self.c=c
        self.min=True
        assert(np.shape(A)==(np.shape(b)[0],np.shape(c)[0])), "Matrix size did not coincide objective and constraint vector."
        self.x=np.zeros(np.shape(A)[1])
        self.B_idx=np.zeros(np.shape(A)[1], dtype=bool)
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

    def set_basis(self, basis, val):
        '''
        Set the current basis solution of problem

        Parameter
        ---------
        basis : array-like
            list of indices to be changed
        val : array-like
            list of corresponding values
        '''
        self.B_idx[basis]=val
    
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

    def iterate_simplex(self):
        '''
        Perform a single simplex iteration.
        '''

        



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
    aux_prob.set_basis(np.arange(n,n+m),True)     

    return aux_prob

def blands_rule(p:Problem):
    '''
    Gives the entering and leaving basis with Bland's rule.
    i.e. minimal index of non-basis that has negative reduced cost.

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
    leaving_idx=np.min(x_idx[p.B_idx])

    return (enter_idx, leaving_idx)