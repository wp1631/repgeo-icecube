from cv import utils


def test_getGaborKernel():
    for s in range(1, 22, 2):
        test_shape = (s, s)
        res = utils.getGaborKernel(test_shape, 2, 10, 1, 1, 0)
        assert res.shape == test_shape
