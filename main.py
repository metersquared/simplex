import numpy as np
from numpy import linalg as la
import simplex as splx

def data_generator(group_num):
    '''
    Generate random data 

    Parameters
    ----------
    group_num : int
        Group number

    Return
    ------
    tuple
        tuples of c (objective vector), b (b vector), A (matrix) for a given LP. 
    '''

    my_group = group_num

    #function to get a seed for your group # DO NOT CHANGE THIS FUNCTION !
    def get_seed_for_my_group(group_number):
        return list(range(1,41))[group_number]


    #seed the pseudo-random number generator with your group number
    seed = get_seed_for_my_group(my_group)
    np.random.seed(seed)

    #generate the data
    c= np.random.randint(-20,21,5)
    b= np.random.randint(-20,21,8)
    A= np.random.randint(-20,21,(8,5))
    
    return c,b,A

if __name__ == '__main__':
    c,b,A= data_generator(17)

    """ A=np.array([[1, 2, 3, 0],
                [-1, 2, 6, 0],
                [0, 4, 9, 0],
                [0, 0, 3, 1]])
    b=np.array([3,2,5,1])
    c=np.array([1,1,1,0]) """
    p1=splx.Problem(A,b,c)
    p1.set_max()
    splx.multiphase_simplex(p1,splx.blands_rule)
    """
    pa=splx.auxillary_problem(p1)
    print(pa)
    pa.sync()
    print(pa.r_cost)
    
    splx.optimize(pa,splx.blands_rule)

    A=np.array([[1, 0, 1, 0],
                [1, -1, 0, 1]])
    b=np.array([7,8])
    c=np.array([-5,-4,0,0])
    
    p2=splx.Problem(A,b,c)
    p2.set_x(np.array([0,0,7,8]))
    p2.set_basis(np.array([0,1]),np.array([2,3]))
    
    print(p2)
    splx.optimize(p2,splx.blands_rule)
    """
    """print(pa) """