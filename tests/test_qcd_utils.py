import numpy as np
import pytest
from showerpy.qcd_utils import (CF, CA, beta0, n_active_flavors, alpha_gauge_1loop,
                        alpha_em, alpha_s, SplittingFunction,
                        F2FG_splitting_function, G2GG_splitting_function,
                        GaugeTheorySplittingFunction)


def test_CF():
    assert CF(2) == .75
    assert CF(3) == 4/3


def test_CA():
    assert CA(2) == 2
    assert CA(3) == 3


def test_beta0():
    # QCD beta function below the top mass and above the bottom mass
    assert beta0(3, 5) == 0.6100939485189322


def test_n_active_flavors():
    masses = np.array([0.0023, 0.0048, 0.095, 1.275, 4.18])
    assert n_active_flavors(1.0, masses) == 3
    assert n_active_flavors(0.1, masses) == 3
    assert n_active_flavors(0.01, masses) == 2
    assert n_active_flavors(1.3, masses) == 4


def test_GaugeTheorySplittingFunction():
    ll_split_fn = GaugeTheorySplittingFunction(accuracy='LL')
    assert ll_split_fn(0.1, split_type='F2FG') == 26.6666666666666640
    assert ll_split_fn(0.5, split_type='G2GG') == 12.0
    assert ll_split_fn(0.2, split_type='G2GG') == 30.0

    reduced_ll_split_fn = GaugeTheorySplittingFunction(accuracy='LL', reduced=True)
    assert reduced_ll_split_fn(0.1, split_type='F2FG') == 29.629629629629626
    assert reduced_ll_split_fn(0.4, split_type='F2FG') == 11.11111111111111
    assert reduced_ll_split_fn(0.3, split_type='G2GG') == 28.57142857142857

    mll_split_fn = GaugeTheorySplittingFunction()
    assert mll_split_fn(0.1, split_type='F2FG') == 48.266666666666660
    assert mll_split_fn(0.5, split_type='G2GG') == 8.00
    assert mll_split_fn(0.2, split_type='G2GG') == 26.180

    reduced_mll_split_fn = GaugeTheorySplittingFunction(reduced=True)
    assert reduced_mll_split_fn(0.1, split_type='F2FG') == 51.259259259259250
    assert reduced_mll_split_fn(0.4, split_type='F2FG') == 14.2222222222222210
    assert reduced_mll_split_fn(0.3, split_type='G2GG') == 20.7314285714285730


# Run the tests
if __name__ == '__main__':
    pytest.main()
