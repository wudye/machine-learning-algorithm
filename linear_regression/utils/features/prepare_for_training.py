import numpy as np
from .generate_polynomials import generate_polynomials
from .generate_sinusoids import generate_sinusoids
from .normalize import normalize


def prepare_for_training(dataset, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    normalize the dataset
    generate the polynomial features
    generate the sinusoid features
    add one column as one feature for the bias term
    """

    num_examples = dataset.shape[0]

    data_processed = np.copy(dataset)

    feature_mean = 0
    feature_deviation = 0
    data_normalized = data_processed

    if normalize_data:
        (
            data_normalized,
            feature_mean,
            feature_deviation
        ) = normalize(data_processed)
        data_processed = data_normalized

    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_processed, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_processed, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, feature_mean, feature_deviation