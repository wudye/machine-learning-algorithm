import numpy as np

def generate_sinusoids(dataset, sinusoid_degree):
    """
    generate the sin(x) for each feature
    get the row number of the dataset
    generate the empty array for the result
    use for loop to generate the sin(x)
    use concatenate to expand the result array
    """
    row_num = dataset.shape[0]
    sinusoids = np.empty((row_num, 0))

    for degree in range(1,  sinusoid_degree + 1):
        sinusoids_feature = np.sin(dataset * degree)
        sinusoids = np.concatenate((sinusoids, sinusoids_feature), axis=1)
    
    return sinusoids

    