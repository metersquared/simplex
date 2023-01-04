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

    def iterate_simplex(self, pivot_rule):
        '''
        Perform a single simplex iteration.

        Parameter
        ---------
        pivot_rule : function
            Function that defines the pivoting rule, must return the index leaving and 
            entering the basis, i.e. (enter_idx, leaving_idx) 

        Returns
        -------
        boolean
            True if iteration passes the boundedness test, otherwise False. Use this as
            a iteration condition.
        '''

        enter_idx, leaving_idx= pivot_rule(self)

        # Gives the pivoting row
        leaving_row=np.where(self.B_idx==leaving_idx)[0][0]
        
        if np.all((self.A[:,enter_idx]).flatten()<=0):
        
            return False
        
        else:
            # Updates the reduced cost and objective value
            cost_factor=(self.r_cost[enter_idx]/self.A[leaving_row, enter_idx])
            self.r_cost = self.r_cost-cost_factor*self.A[leaving_row,:]
            self.obj_val = self.obj_val+cost_factor*self.x[leaving_idx]

            # Updates the constraint matrix and solution with row operations
            for row in np.arange(np.shape(self.A)[0]):
                factor=(self.A[row,enter_idx]/self.A[leaving_row,enter_idx])
                if row!= leaving_row:
                    self.A[row,:]=self.A[row,:]-factor*self.A[leaving_row,:]
                    self.x[self.B_idx[row]]=self.x[self.B_idx[row]]-factor*self.x[leaving_idx]
            
            # Normalize entering basis and eliminate leaving basis. 
            self.x[enter_idx]=self.x[leaving_idx]/self.A[leaving_row,enter_idx]
            self.A[leaving_row,:]=self.A[leaving_row,:]/self.A[leaving_row,enter_idx]
            self.x[leaving_idx]=0
            self.B_idx[leaving_row]=enter_idx
            
            return True

#############
# Optimizer #
#############

def optimize(p:Problem, pivot_rule):
    '''
    Perform a general Simplex optimization. Stops when reduced cost is non-negative (optimal) or pivot column is non-positive (unbounded).

    Parameter
    ---------
    p : Problem
        Problem to be optimized
    pivot_rule : function
        Function that defines the pivoting rule, must return the index leaving and 
        entering the basis, i.e. (enter_idx, leaving_idx) 
    '''
    p.sync()

    while np.any(p.r_cost<0):
        if not p.iterate_simplex(pivot_rule):
            # Checks if iteration is unbounded
            print("--UNBOUNDED--\n")
            return False

    print("--Optimal value--")
    print(p.obj_val)
    print("--Solution--")
    print(p.x)
    print("\n")
    return True

###########
# Phase 1 #
###########

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
    aux_prob.artificial_var=np.arange(n,n+m)
    aux_prob.set_basis(np.arange(m),aux_prob.artificial_var) 

    return aux_prob

def phase_1(p:Problem):
    '''
    Perform a Phase 1 Simplex optimization

    Parameter
    ---------
    p : Problem
        Problem to which one wants to generate auxillary problem
    '''
    print("--Phase 1--\n")
    pa = auxillary_problem(p)
    pa.sync()
   
    while optimize(pa,blands_rule):
        while has_artificial_basis(pa):
            if has_nonzero_row(pa):
                pa.iterate_simplex(artificial_elimination)
        if not has_artificial_basis(pa):
            break

    print(pa.B_idx)
    print(pa.x[pa.B_idx])        

    if not np.isclose(pa.obj_val,0):
        print("INFEASIBLE")

    
def has_artificial_basis(pa:Problem):
    '''
    Check if basis is artificial (Basis contains artificial variable).

    Parameter
    ---------
    p,pa : Problem
        Problem to which one wants to generate auxillary problem

    Returns
    -------
    boolean
        True if it has artificial basis, otherwise False.
    '''
    return len(pa.B_idx[pa.B_idx==pa.artificial_var])>0

def has_nonzero_row(pa:Problem):
    '''
    Check if artificial basis contains nonzero pivot element.

    Parameter
    ---------
    p,pa : Problem
        Problem to which one wants to generate auxillary problem

    Returns
    -------
    boolean
        True if it has artificial basis, otherwise False.
    '''

    # Find the artificial basis to choose as leaving index
    leaving_idx=np.min(pa.B_idx[pa.B_idx==pa.artificial_var])
    leaving_row=np.where(pa.B_idx==leaving_idx)[0][0]

    # List all possible non-artifical variables
    x_idx=np.arange(np.min(pa.artificial_var))

    # Calculate the entering basis by finding the smallest index where pivot element is non-zero.
    enter_idxs=[i for i in x_idx[pa.A[leaving_row,x_idx]!=0] if i not in pa.B_idx]

    if len(enter_idxs)==0:
        pa.A=np.delete(pa.A,leaving_row,0)
        pa.B_idx=np.delete(pa.B_idx,leaving_row)
        pa.x[leaving_idx]=0
        return False
    else:
        return True

##################
# Pivoting rules #
##################


def blands_rule(p:Problem):
    '''
    Gives the entering and leaving basis with Bland's rule.
    
    i.e. Minimal index of non-basis that has negative reduced cost enters.
    Minimal index of basis that are candidates exit.

    Parameter
    ---------
    p : Problem
        LP problem

    Returns
    -------
    (int,int)
        Entering and leaving basis.
    '''

    # List all possible index of variables
    x_idx=np.arange(p.n)

    # Calculate the entering basis by finding the smallest index where the reduced cost is negative.
    enter_idx=np.min([i for i in x_idx[(p.r_cost<0)] if i not in p.B_idx])

    # Calculate the leaving basis by finding minimum x/u where u>0
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = p.x[p.B_idx]/(p.A[:,enter_idx]).flatten()
    leaving_idx=np.min(p.B_idx[theta==np.min(theta,initial=np.inf,where=(p.A[:,enter_idx]).flatten()>0)])

    return (enter_idx, leaving_idx)

def artificial_elimination(p:Problem):
    '''
    Gives the entering and leaving basis for eliminating artifcial variable in Phase 1.

    i.e. Index where pivot element is non-zero enters. Artificial variable exits.

    Parameter
    ---------
    p : Problem
        LP auxillary problem

    Returns
    -------
    (int,int)
        Entering and leaving basis.
    '''

    # Find the artificial basis to choose as leaving index
    leaving_idx=np.min(p.B_idx[p.B_idx==p.artificial_var])
    leaving_row=np.where(p.B_idx==leaving_idx)[0][0]

    # List all possible non-artifical variables
    x_idx=np.arange(np.min(p.artificial_var))

    # Calculate the entering basis by finding the smallest index where pivot element is non-zero.
    enter_idxs=[i for i in x_idx[p.A[leaving_row,x_idx]!=0] if i not in p.B_idx]
    if len(enter_idxs)>0:
        enter_idx=np.min(enter_idxs)
        return (enter_idx, leaving_idx)
    else:
        p.A=np.delete(p.A,leaving_row,0)
        p.B_idx=np.delete(p.B_idx,leaving_row)
        p.x[leaving_idx]=0