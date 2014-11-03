import numpy as np
from numpy import dot
from scipy.optimize import fmin_cg


# Each column vector of X is an observation
def prepend_ones(X):
    return np.append( np.ones(X.shape[1]).reshape(1,-1) , X, 0)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


# num_units is a list containing number of units in each layer
def init_epsilons(num_units):
    assert len(num_units) > 1

    epsilons = []
    for idx in range(len(num_units) - 1):
        epsilon = np.sqrt(6) / np.sqrt(num_units[idx] + num_units[idx+1])
        epsilons.append(epsilon)

    return epsilons


def random_matrix(rows, cols, epsilon):
    return np.random.rand(rows, cols) * 2 * epsilon - epsilon


def init_thetas(layer_sizes):
    assert len(layer_sizes) > 1

    epsilons = init_epsilons(layer_sizes)

    num_cols = []
    num_rows = []
    for idx in range(len(layer_sizes) - 1):
        num_cols.append(layer_sizes[idx] + 1)
        num_rows.append(layer_sizes[idx+1])

    return map(random_matrix, num_rows, num_cols, epsilons)


# input hasn't been prepended with 1
def feedforward(Thetas, X):
    X, As, Zs = feedforward_full(Thetas, X)
    return X

# input hasn't been prepended w/ 1
# As doesn't include result
def feedforward_full(Thetas, X):
    assert X.ndim == 2

    As = []
    Zs = []
    for Theta in Thetas:
        assert Theta.ndim == 2
        X = prepend_ones(X)
        As.append(X)
        X = dot(Theta, X)
        Zs.append(X)
        X = sigmoid(X)

    return (X, As, Zs)

def predict_labels(Thetas, X):
    Y_est = feedforward(Thetas, X)
    return np.argmax(Y_est, axis=0)

# note scaled by num obs
def backprop(Thetas, Zs, Y_est, Y):
    assert Y_est.ndim == 2
    assert Y.ndim == 2, 'Y.ndim = ' + str(Y.ndim)
    assert Y_est.shape == Y.shape
    assert len(Thetas) == len(Zs)
    
    # deltas[i] = grad of cost w/ respect to z in ith layer (after linear
    # combination w/ ith Theta)
    num_obs = Y.shape[1]
    num_layers = len(Thetas)
    deltas = [None] * num_layers
    delta = Y_est - Y
    deltas[-1] = delta

    for lyr in range(num_layers-1, 0, -1):
        derivatives = sigmoid_der(Zs[lyr-1])  # can use As here as in a(1-a)
        prev = dot(Thetas[lyr][:,1:].transpose(), deltas[lyr])
        deltas[lyr-1] = (derivatives * prev)

    return deltas
    # return [ np.sum(d, axis=1)/num_obs for d in deltas ]


def calc_grads(deltas, As, lmbda, Thetas):
    assert len(deltas) == len(As)
    assert all([delta.ndim == 2 for delta in deltas])
    assert all([A.ndim == 2 for A in As])
    assert len(set([A.shape[1] for A in As])) == 1
    # add asserts for Thetas

    num_obs = float(As[0].shape[1])
    grads = []
    for delta, A, Theta in zip(deltas, As, Thetas):
        unregularised = dot(delta, A.transpose())
        regulariser = lmbda * Theta
        regulariser[:,0] = 0.
        grads.append((unregularised + regulariser) / num_obs)

    return grads


# X has not been prepended w/ ones
# Column vectors of X and Y are observations, note Y will be zero except for
# exactly one 1 in each column
def calc_cost(X, Y, lmbda, Thetas):
    
    assert Y.ndim == 2, 'Y shape wrong'
    assert X.ndim == 2, 'X shape wrong'
    assert X.shape[1] == Y.shape[1]
    assert all(map(lambda Theta: Theta.ndim == 2, Thetas))
    # assert X.shape[1] == Thetas.shape[0]
    
    m = X.shape[1]  # num cols/observations
    feedforw = feedforward(Thetas, X)
    costs = - Y * np.log(feedforw) - (1 - Y) * np.log(1 - feedforw)
    assert costs.ndim == 2
    cost = np.sum(costs) / float(m)
    regulariser = np.sum([ np.sum(Theta[:,1:]*Theta[:,1:]) for Theta in Thetas ])
    regulariser *= float(lmbda) / float(2*m)
    return cost + regulariser


def approximate_grad(X, Y, lmbda, Thetas, epsilon):

    approx_grads = []
    for Theta in Thetas:
        grad = np.empty(Theta.shape)

        for row_idx, row in enumerate(Theta):
            for col_idx, theta in enumerate(row):
                Theta[row_idx, col_idx] = theta + epsilon
                cost_plus = calc_cost(X, Y, lmbda, Thetas)
                Theta[row_idx, col_idx] = theta - epsilon
                cost_minus = calc_cost(X, Y, lmbda, Thetas)
                grad[row_idx, col_idx] = (cost_plus - cost_minus) / (2*epsilon)
                Theta[row_idx, col_idx] = theta

        approx_grads.append(grad)

    return approx_grads


def flatten(Thetas):
    flattened_thetas = [Theta.reshape(-1) for Theta in Thetas]
    return np.concatenate(flattened_thetas)


def unflatten(theta_arr, layer_sizes):
    assert len(theta_arr) == np.sum(np.array(layer_sizes[1:]) * 
                                    (np.array(layer_sizes[:-1]) + 1))

    Thetas = []
    num_cols = layer_sizes[0] + 1
    curr_mat_start = 0
    for num_rows in layer_sizes[1:]:
        mat_size = num_rows * num_cols
        Thetas.append(theta_arr[curr_mat_start:curr_mat_start + mat_size]
                      .reshape(num_rows, num_cols))
        curr_mat_start += mat_size
        num_cols = num_rows + 1

    return Thetas


def get_obj(layer_sizes, X, Y, lmbda):

    def obj(theta_arr):
        return calc_cost(X, Y, lmbda, unflatten(theta_arr, layer_sizes))

    return obj


def get_obj_grad(layer_sizes, X, Y, lmbda):

    def obj_grad(theta_arr):

        Thetas = unflatten(theta_arr, layer_sizes)
        Y_est, As, Zs = feedforward_full(Thetas, X)
        deltas = backprop(Thetas, Zs, Y_est, Y)
        grads = calc_grads(deltas, As, lmbda, Thetas)

        return flatten(grads)

    return obj_grad



# layer_sizes: list of ints
# features:  2d array, observations along columns, not prepended w/ 1s
# classes: 2d array of known labels, array of 0s and one 1 per column.
# lmbda: regularisation parameter
# init_thetas: list of 2d arrays. Avoid symmetry
def train(layer_sizes, features, classes, lmbda, init_thetas, max_iter=None):
    assert features.ndim == 2
    assert classes.ndim == 2

    f = get_obj(layer_sizes, features, classes, lmbda)
    f_prime = get_obj_grad(layer_sizes, features, classes, lmbda)
    min_thetas = fmin_cg(f, flatten(init_thetas), f_prime, maxiter=max_iter)
    return unflatten(min_thetas, layer_sizes)


