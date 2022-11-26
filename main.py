import numpy as np
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
    splx.simplex(c,A.T,b)