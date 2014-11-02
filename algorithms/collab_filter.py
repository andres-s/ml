import numpy as np
from numpy import dot
from scipy.optimize import fmin_cg

def _diff_pred_actual(Y, R, X, Theta):
    return R * (dot(X, Theta.transpose()) - Y)    

def calc_cost(Y, R, X, Theta, lmbda):

    assert (Y.ndim, R.ndim, X.ndim, Theta.ndim) == (2, 2, 2, 2)
    assert X.shape[1] == Theta.shape[1]
    assert Y.shape == R.shape
    assert X.shape[0] == Y.shape[0] # num movies
    assert Theta.shape[0] == Y.shape[1] # num users

    R = R.astype(np.float64)
    
    sqr_diff = _diff_pred_actual(Y, R, X, Theta)
    sqr_diff = sqr_diff * sqr_diff
    sqr_diff = np.sum(sqr_diff) / 2
    
    regulariser = np.sum(X*X) + np.sum(Theta*Theta)
    regulariser *= lmbda / 2

    return sqr_diff + regulariser


def calc_X_grad(Y, R, X, Theta, lmbda):
    diff = _diff_pred_actual(Y, R, X, Theta)
    unreg = dot(diff, Theta)
    reg = lmbda * X
    return unreg + reg


def calc_Theta_grad(Y, R, X, Theta, lmbda):
    diff = _diff_pred_actual(Y, R, X, Theta)
    unreg = dot(diff.transpose(), X)
    reg = lmbda * Theta
    return unreg + reg


def random_matrix(rows, cols, epsilon):
    return np.random.rand(rows, cols) * 2 * epsilon - epsilon


def flatten(X, Theta):
    return np.concatenate([X.reshape(-1), Theta.reshape(-1)])

def unflatten(param_arr, X_rows, X_cols, Theta_rows, Theta_cols):
    X_size = X_rows * X_cols
    X = param_arr[:X_size].reshape(X_rows, X_cols)
    Theta = param_arr[X_size:].reshape(Theta_rows, Theta_cols)
    return (X, Theta)


def get_obj(Y, R, num_features, lmbda):

    def obj(param_arr):
        (X, Theta) = unflatten(param_arr, Y.shape[0], num_features,
                               Y.shape[1], num_features)
        return calc_cost(Y, R, X, Theta, lmbda)

    return obj

def get_obj_grad(Y, R, num_features, lmbda):

    def obj_grad(param_arr):
        (X, Theta) = unflatten(param_arr, Y.shape[0], num_features,
                               Y.shape[1], num_features)
        X_grad = calc_X_grad(Y, R, X, Theta, lmbda)
        Theta_grad = calc_Theta_grad(Y, R, X, Theta, lmbda)
        return flatten(X_grad, Theta_grad)

    return obj_grad

# num_features: number of derived features
# Y: movie ratings
# R: matrix of movie rated by user flags
# lmbda: regularisation parameter
# epsilon: parameter initialisation constraint
def train(Y, R, num_features, lmbda, epsilon, max_iter=None):
    assert (Y.ndim, R.ndim) == (2, 2)
    assert Y.shape == R.shape

    init_X = random_matrix(Y.shape[0], num_features, epsilon)
    init_Theta = random_matrix(Y.shape[1], num_features, epsilon)
    f = get_obj(Y, R, num_features, lmbda)
    f_prime = get_obj_grad(Y, R, num_features, lmbda)
    min_thetas = fmin_cg(f, flatten(init_X, init_Theta), f_prime, maxiter=max_iter)
    return unflatten(min_thetas, init_X.shape[0], init_X.shape[1],
                                 init_Theta.shape[0], init_Theta.shape[1])

def predict(X, Theta):
    return dot(X, Theta.transpose())
    