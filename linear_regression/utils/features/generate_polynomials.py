import numpy as np
from .normalize import normalize

def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    generate the polynomial features -> x1, x2,  x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3, ...
    split the dataset 
    check the splited dataset 
    create the empty array for the result
    use for loop to generate the polynomial features
    use concatenate to expand the result array
    if normalize_data is True, normalize the result array
    """

    features_split = np.array_split(dataset, 2, axis=1)
    dataset_split_1 = features_split[0]
    dataset_split_2 = features_split[1]

    (num_exaples_1, num_features_1) = dataset_split_1.shape
    (num_exaples_2, num_features_2) = dataset_split_2.shape

    if num_exaples_1 != num_exaples_2:
        raise ValueError("The number of examples in the two splits must be the same.")
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError("The number of features in the splits must be greater than zero.")
    
    if num_features_1 == 0:
        dataset_split_1 = dataset_split_2
    elif num_features_2 == 0:
        dataset_split_2 = dataset_split_1

    num_features = min(num_features_1, num_features_2)
    dataset_1 = dataset_split_1[:, :num_features]
    dataset_2 = dataset_split_2[:, :num_features]

    polynomials = np.empty((num_exaples_1,0))

    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 **(i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials