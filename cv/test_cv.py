from cv import utils


def test_getGaborKernel():
    test_shape = (11, 11)
    res = utils.getGaborKernel(test_shape, 2, 10, 1, 1, 0)
    assert res.shape == test_shape
