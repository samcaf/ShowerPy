import pytest
from showerpy.vector_utils import angle
from showerpy.parton import Parton
from showerpy.shower_algorithm import ShowerAlgorithm


def test_ShowerAlgorithm_init():
    cutoff_scale = 100.0
    split_soft = True
    use_veto = False

    algorithm = ShowerAlgorithm(cutoff_scale, split_soft=split_soft, use_veto=use_veto)

    assert algorithm.cutoff_scale == cutoff_scale
    assert algorithm.split_soft == split_soft
    assert algorithm.use_veto == use_veto


def test_ShowerAlgorithm_call():
    # Define a dummy ShowerAlgorithm class for testing
    z0 = 0.45
    theta0 = 0.3
    class DummyShowerAlgorithm(ShowerAlgorithm):
        def generation_pdf(self, momentum_fraction, theta, **kwargs):
            return 1.0

        def true_pdf(self, momentum_fraction, theta, **kwargs):
            return 1.0

        def get_scale(self, momentum_fraction, theta):
            return 1.0

        def random_emission_scale(self, initial_scale, **kwargs):
            return 1.0

        def random_z_and_theta(self, scale):
            return z0, theta0

    algorithm = DummyShowerAlgorithm(5.0)

    seed_parton = Parton([100.0, 50.0, 30.0])
    initial_scale = 10.0

    jet = algorithm(seed_parton, initial_scale)

    assert len(jet) == 3
    assert jet[0].momentum == seed_parton.momentum
    assert jet[0].daughters[0].momentum.mag() == pytest.approx((1-z0) *
                                                   seed_parton.momentum.mag())
    assert jet[0].daughters[1].momentum.mag() == pytest.approx(z0 *
                                                   seed_parton.momentum.mag())
    assert angle(jet[0].daughters[0].momentum,
                 jet[0].daughters[1].momentum) == pytest.approx(theta0)


# Run the tests
if __name__ == '__main__':
    pytest.main()
