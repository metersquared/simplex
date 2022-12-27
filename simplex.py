import numpy as np

class Problem:
    def __init__(self, A, b, c):
        self.A=A
        self.b=b
        self.c=c
        assert(np.shape(A)==(np.shape(b)[0],np.shape(c)[0])), "Matrix size did not coincide objective and constraint vector."
        self.x=np.zeros(np.shape(A)[1])

    def __str__(self) -> str:
        return f"===LP Solver===\nObjective : {self.c}\nConstraints :\n--A--\n{self.A}\n--b--\n{self.b}\n--x--\n{self.x}\n============"

    def transform_problem(self):
        '''
        Transforms constraints in problem such that b>=0 

        Parameter
        ---------
        A : ndarray(m,n)
            constraint matrix
        b : ndarray(m,)
            constraint vector
        '''
        A=self.A
        b=self.b

        assert(np.shape(A)[0]==np.shape(b)[0])
        for i,bi in enumerate(b):
            if bi<0:
                A[i,:]=-A[i,:]
                b[i]=-bi
    
    def auxillary_problem(self):
        '''
        Generate auxillary problem
        '''
        A=self.A
        b=self.b
        c=self.c
        m,n=np.shape(A)

        aux_prob = Problem(
            np.append(A,np.diag(np.ones(m)),axis=1),
            b,
            np.append(np.zeros(n),np.ones(m))
            )
        return aux_prob

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