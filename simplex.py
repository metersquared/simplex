import numpy as np

class Problem:
    def __init__(self, A, b, c):
        self.A=A
        self.b=b
        self.c=c
        self.x=np.zeros(np.shape(A)[1])

    def __str__(self) -> str:
        return f"===LP Solver===\nObjective : {self.c}\nConstraints :\n--A--\n{self.A}\n--b--\n{self.b}\n--x--\n{self.x}\n============"

    def transform_problem(self):
        A=self.A
        b=self.b
        '''
        Transforms constraints in problem such that b>=0 

        Parameter
        ---------
        A : ndarray(m,n)
            constraint matrix
        b : ndarray(m,)
            constraint vector
        '''
        assert(np.shape(A)[0]==np.shape(b)[0])
        for i,bi in enumerate(b):
            if bi<0:
                A[i,:]=-A[i,:]
                b[i]=-bi

def simplex_method(c,A,b):
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


def phase2():
    return