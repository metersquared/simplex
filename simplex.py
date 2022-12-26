import numpy as np

def simplex(c,A,b):
    '''
    Run the simplex algorithm on a LP of the form 
    min c'x
    subject to Ax<=b

    Parameter
    ---------
    c : ndarray(n,)
        objective vector
    A : ndarray(m,n)
        constraint matrix
    b : ndarray(m,)
        constraint vector
    '''
    assert(np.shape(A)==(np.shape(b)[0],np.shape(c)[0])), "Matrix size did not coincide objective and constraint vector."

    isFeasible=False

    c,A,b=phase1(c,A,b)

    if(isFeasible):
        c,A,b=phase2(c,A,b)
    
    return 

def phase1():
    return

def transform_problem(A,b):

def phase2():
    return