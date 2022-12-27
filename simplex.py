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

    def __str__(self) -> str:
        if self.min:
            return f"===LP Simplex Solver===\nObjective :\nMin {self.c}\nConstraints :\n--A--\n{self.A}\n--b--\n{self.b}\n--x--\n{self.x}\n=======================\n"
        else:
            return f"===LP Simplex Solver===\nObjective :\nMax {-self.c}\nConstraints :\n--A--\n{self.A}\n--b--\n{self.b}\n--x--\n{self.x}\n=======================\n"

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
    
    def set_x(self,x):
        assert(np.shape(self.x)==np.shape(x))
        self.x = x


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

        aux_prob.set_x(np.append(np.zeros(n),b))

        return aux_prob