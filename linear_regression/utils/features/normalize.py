import numpy as np

def normalize(features):
    """
    use normal distribution to normalize the featuers
    return the normalized features after x = (x - mean) / std
    return the mean and the std
    if only one sample do not subtract the mean
    check the std if it is zero, if it is zero set it to 1 to avoid division by zero
    """

    features_normalized = np.copy(features).astype(float)

    features_mean = np.mean(features_normalized, axis=0)
    features_deviation = np.std(features_normalized, axis=0)

    if features.shape[0] > 1:
        features_normalized -= features_mean

    features_deviation[features_deviation == 0] =1

    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation