import pytest
from showerpy.qcd_utils import Z_MASS, ALPHA_S_ZMASS
from showerpy.qcd_utils import alpha_gauge_1loop
from showerpy.gaugetheoryshower import AngularityShowerAlgorithm, GaugeTheoryShowerAlgorithm, qcd_shower_algorithm

def test_AngularityShowerAlgorithm_init():
    algorithm = AngularityShowerAlgorithm(cutoff_scale=1.0, beta=2, alpha=1.5, split_soft=False, use_veto=True)
    assert algorithm.cutoff_scale == 1.0
    assert algorithm.beta == 2
    assert algorithm.alpha == 1.5
    assert algorithm.split_soft == False
    assert algorithm.use_veto == True

def test_GaugeTheoryShowerAlgorithm_init():
    algorithm = GaugeTheoryShowerAlgorithm(cutoff_scale=1.0,
                                           beta=2, alpha=1.5,
                                           alpha0=ALPHA_S_ZMASS,
                                           mu0=Z_MASS,
                                           NC=20,
                                           NF=25,
                                           accuracy='LL',
                                           split_soft=False,
                                           use_veto=True,
                                           coupling=None,
                                           splitting_function=None)
    assert algorithm.cutoff_scale == 1.0
    assert algorithm.beta == 2
    assert algorithm.alpha == 1.5
    assert algorithm.split_soft == False
    assert algorithm.use_veto == True


def test_GaugeTheoryShowerAlgorithm_random_z_and_theta():
    algorithm = GaugeTheoryShowerAlgorithm(cutoff_scale=1.0,
                                           beta=2, alpha=1.5,
                                           alpha0=ALPHA_S_ZMASS,
                                           mu0=Z_MASS,
                                           NC=20,
                                           NF=25,
                                           accuracy='LL',
                                           split_soft=False,
                                           use_veto=True,
                                           coupling=None,
                                           splitting_function=None)
    scale = 0.4
    z, theta = algorithm.random_z_and_theta(scale)
    assert z >= 0 and z <= 0.5
    assert theta >= 0 and theta <=1.0

def test_GaugeTheoryShowerAlgorithm_random_emission_scale():
    algorithm = GaugeTheoryShowerAlgorithm(cutoff_scale=1.0,
                                           beta=2, alpha=1.5,
                                           alpha0=ALPHA_S_ZMASS,
                                           mu0=Z_MASS,
                                           NC=20,
                                           NF=25,
                                           accuracy='LL',
                                           split_soft=False,
                                           use_veto=True,
                                           coupling=None,
                                           splitting_function=None)
    initial_scale = 0.5
    with pytest.raises(KeyError):
        algorithm.random_emission_scale(initial_scale)

def test_qcd_shower_algorithm():
    algorithm = qcd_shower_algorithm()
    assert isinstance(algorithm, GaugeTheoryShowerAlgorithm)
    assert algorithm.cutoff_scale == 1e-8
    assert algorithm.beta == 2
    assert algorithm.alpha == 1.5
    assert algorithm.split_soft == False
    assert algorithm.use_veto == True
    assert algorithm.coupling is not None
    assert algorithm.splitting_function is not None


# Run the tests
if __name__ == '__main__':
    pytest.main()
