import unittest as ut

import numpy as np

from neural_networks import flatten
from neural_networks import unflatten
from neural_networks import feedforward_full
from neural_networks import calc_cost
from neural_networks import backprop
from neural_networks import calc_grads
from neural_networks import approximate_grad
from neural_networks import NeuralNetwork


class NumpyCustomAssertions:

    def assertArrayEqual(self, x, y, err_msg='', verbose=True):
        np.testing.assert_array_equal(x, y, err_msg, verbose)

    def assertAllClose(self, actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
        np.testing.assert_allclose(actual, desired, rtol, atol, err_msg, verbose)

    def assertArrayListClose(self, actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
        self.assertEqual(len(actual), len(desired))
        for act, des in zip(actual, desired):
            self.assertAllClose(act, des, rtol, atol, err_msg, verbose)


class FlattenUnflattenTestCase(ut.TestCase, NumpyCustomAssertions):

    def setUp(self):
        pass

    def test_flatten_unflatten(self):
        list_of_arrs = [
            np.array([[1, 2, 3],
                      [4, 5, 6]]),
            np.array([[7, 8, 9]]),
            np.array([[10, 11],
                      [12, 13],
                      [14, 15],
                      [16, 17],
                      [18, 19]]),
        ]
        flat_unflat = unflatten(flatten(list_of_arrs), [2, 2, 1, 5])
        self.assertEqual(len(list_of_arrs), len(flat_unflat))
        for orig, new in zip(list_of_arrs, flat_unflat):
            self.assertArrayEqual(orig, new)

    def tearDown(self):
        pass

_Thetas = [
    # Theta0
    np.array([[-3.615850357, -0.3572016499, 1.019212194, 1.738035266, -0.8919005405, 2.435389833, 0.206440591, 1.565399323, 1.393486856],
              [-0.1730933504, 1.947875383, -0.04220509044, -0.2361687846, -1.109454465, 1.089258652, -0.4171368526, 2.165621746, 1.33556253],
              [1.581242334, -0.1385191813, -0.6693216397, -1.354666321, -3.195696917, 3.370635165, -1.336427658, 0.4522263749, 1.471862393],
              [-1.825365381, -3.079400998, -0.5362374188, 0.9149184919, -2.26237002, -1.990985896, -1.746806614, 1.194315272, 2.311141334]]),
    # Theta1
    np.array([[-2.013329942, -0.2243117431, 0.3268978913, -1.916844198, 1.089454646],
              [2.206011981, 2.154621951, -0.1785171128, 3.465666206, -0.7763444591],
              [-0.1716802431, -0.3535945295, 0.9245926192, 1.867739293, 0.03214243266]]),
]

_layer_sizes = [ Theta.shape[1] - 1 for Theta in _Thetas ]
_layer_sizes.append(_Thetas[-1].shape[0])

# observations go down columns, not prepended w/ 1s
_X = np.array([
    [1, 2, 3, 4, 0.5, 0.6, 0.7, 0.8],
    [-3.186680868, -1.386888978, -1.066600131, 0.7693032233, -0.1276599716, 2.234706901, 0.414228841, -0.3478238987],
    [-1.354928569, 1.130253162, 1.785750029, 0.7441751385, -1.564252874, -0.8919228085, 1.180114288, 1.176998497]
]).transpose()

_Zs = [ # observations go down columns
    # Z0
    np.array([
        [3.2640043, -0.5772199019, -14.36519739, -11.64060898],
        [-6.116849521, -7.562136943, -1.804082108, 2.056521947],
        [-0.04614608711, -1.311580486, -5.59930739, 10.49319563],
    ]).transpose(),
    # Z1
    np.array([
        [-2.111828987, 4.217091178, -0.1797934968],
        [-1.318693105, 2.012238546, 0.1205336793],
        [-0.9711659565, 2.457039401, -0.1091521398],
    ]).transpose(),
]

_As = [
    # A0
    np.array([
        [1, 1, 2, 3, 4, 0.5, 0.6, 0.7, 0.8],
        [1, -3.186680868, -1.386888978, -1.066600131, 0.7693032233, -0.1276599716, 2.234706901, 0.414228841, -0.3478238987],
        [1, -1.354928569, 1.130253162, 1.785750029, 0.7441751385, -1.564252874, -0.8919228085, 1.180114288, 1.176998497]
    ]).transpose(),
    # A1
    np.array([
        [1, 0.9631730897, 0.3595725448, 0.0000005771301855, 0.000008801241397],
        [1, 0.002200540018, 0.0005194933383, 0.1413548787, 0.8866049682],
        [1, 0.488465525, 0.2122224918, 0.003686783114, 0.9999722763],
    ]).transpose(),
]

_Y_est = np.array([
    [0.1079524114, 0.9854726913, 0.455172318],
    [0.211035809, 0.8820760702, 0.5300969903],
    [0.2746481634, 0.921074707, 0.4727390257],
]).transpose()

_Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

_cost = 3.9307932

_regulariser = 18.76378396

_lmbda = 0.1

_regularised_cost = 5.807171595

_deltas = [
    # deltas0
    np.array([0.2008999048, -0.03481036135, 0.009215229742, 0.01133489901]),
    # deltas1
    np.array([-0.1354545388, 0.5962078228, 0.1526694447]),
]

_obs_deltas = [
    #0
    np.array([
        [0.07670435047, -0.01075022477, 0.000003448569527, -0.00001515808013],
        [-0.001073385881, 0.0003012339134, 0.02146799745, 0.03403185363],
        [0.5270687497, -0.09398209319, 0.006174243207, -0.00001199853429],
    ]).transpose(),
    #1
    np.array([
        [-0.8920475886, 0.9854726913, 0.455172318],
        [0.211035809, -0.1179239298, 0.5300969903],
        [0.2746481634, 0.921074707, -0.5272609743],
    ]).transpose(),
]

_grads = [
    #grads0
    np.array([
        [0.2008999048, -0.2113385393, 0.250206163, 0.3908236532, 0.2327410342, -0.2619932009, -0.142160244, 0.22508326, 0.2273653185],
        [-0.03481036135, 0.03854295397, -0.04271392852, -0.06680016535, -0.03756943203, 0.04719939715, 0.02601593572, -0.03943666285, -0.03977391285],
        [0.009215229742, -0.02559128889, -0.007596124665, -0.003953922729, 0.007041304069, -0.004132319113, 0.01415660095, 0.005393796775, -0.00006574958078],
        [0.01133489901, -0.03614918593, -0.01574742675, -0.01212176005, 0.008703751122, -0.001444438589, 0.02535094169, 0.004690734996, -0.003954446911],
    ]),
    #grads1
    np.array([
        [-0.1354545388, -0.2415252267, -0.08745322406, 0.01028099818, 0.1539126982],
        [0.5962078228, 0.4662781736, 0.1832534774, -0.004424250455, 0.2721686342],
        [0.1526694447, 0.06067580629, 0.01734873757, 0.02432938719, -0.01908524178],
    ]),
]

_regularised_grads = [
    #0
    np.array([
        [0.2008999048, -0.223245261, 0.2841799028, 0.4487581621, 0.2030110161, -0.1808135398, -0.135278891, 0.2772632375, 0.2738148804],
        [-0.03481036135, 0.1034721334, -0.04412076487, -0.07467245817, -0.07455124753, 0.08350801889, 0.01211137397, 0.0327507287, 0.004744838137],
        [0.009215229742, -0.03020859493, -0.02990684599, -0.04910946677, -0.09948192649, 0.1082221864, -0.03039098766, 0.02046800927, 0.04899633018],
        [0.01133489901, -0.1387958858, -0.03362200738, 0.01837552301, -0.06670858287, -0.06781063512, -0.03287594543, 0.04450124405, 0.07308359756],
    ]),
    #1
    np.array([
        [-0.1354545388, -0.2490022848, -0.07655662769, -0.05361380843, 0.1902278531],
        [0.5962078228, 0.5380989053, 0.177302907, 0.1110979564, 0.2462904856],
        [0.1526694447, 0.04888932197, 0.04816849155, 0.08658736363, -0.01801382735],
    ]),
]

class FeedforwardFullTestCase(ut.TestCase, NumpyCustomAssertions):

    def setUp(self):
        pass

    def test_feedforward_full(self):
        Y_est, As, Zs = feedforward_full(_Thetas, _X)

        self.assertArrayListClose(As, _As)
        self.assertArrayListClose(Zs, _Zs)
        self.assertAllClose(Y_est, _Y_est)

    def tearDown(self):
        pass


class DecisionFunctionTestCase(ut.TestCase, NumpyCustomAssertions):

    def setUp(self):
        self.neural_network = NeuralNetwork(_layer_sizes)
        self.neural_network._Thetas = _Thetas


    def test_feedforward_full(self):
        Y_est = self.neural_network.decision_function(_X.transpose())
        self.assertAllClose(Y_est, _Y_est)

    def tearDown(self):
        pass


class CalcCostTestCase(ut.TestCase, NumpyCustomAssertions):

    def test_calc_cost(self):
        cost = calc_cost(_X, _Y, 0, _Thetas)
        self.assertAllClose(cost, _cost)

    def test_calc_reg_cost(self):
        cost = calc_cost(_X, _Y, _lmbda, _Thetas)
        self.assertAllClose(cost, _regularised_cost)        


class BackpropTestCase(ut.TestCase, NumpyCustomAssertions):

    def test_backprop(self):
        deltas = backprop(_Thetas, _Zs, _Y_est, _Y)
        self.assertArrayListClose(deltas, _obs_deltas)

class CalcGradsTestCase(ut.TestCase, NumpyCustomAssertions):

    def test_calc_grads(self):
        grads = calc_grads(_obs_deltas, _As, 0, _Thetas)
        self.assertArrayListClose(grads, _grads)

    def test_compare_with_numeric(self):
        grads = calc_grads(_obs_deltas, _As, 0, _Thetas)
        approx_grads = approximate_grad(_X, _Y, 0, _Thetas, 1e-04)
        self.assertArrayListClose(grads, approx_grads, rtol=1e-04)

    def test_calc_reg_grads(self):
        grads = calc_grads(_obs_deltas, _As, _lmbda, _Thetas)
        self.assertArrayListClose(grads, _regularised_grads)

    def test_compare_with_numeric_reg(self):
        grads = calc_grads(_obs_deltas, _As, _lmbda, _Thetas)
        approx_grads = approximate_grad(_X, _Y, _lmbda, _Thetas, 1e-04)
        self.assertArrayListClose(grads, approx_grads, rtol=1e-04)


# suite = ut.TestLoader().loadTestsFromTestCase(FlattenUnflattenTestCase)


if __name__ == '__main__':
    ut.main()
    # ut.TextTestRunner(verbosity=2).run(suite)