import random
import numpy as np

from showerpy.parton import Parton
from showerpy.shower_algorithm import ShowerAlgorithm

from showerpy.qcd_utils import alpha_gauge_1loop,\
    alpha_s, quark_masses, ALPHA_S_ZMASS, Z_MASS,\
    GaugeTheorySplittingFunction


# =====================================
# Angularity-based shower algorithm
# =====================================
# Still needs the `true_pdf` method to be defined
class AngularityShowerAlgorithm(ShowerAlgorithm):
    """
        TODO: AngularityShowerAlgorithm docstring
    """
    def __init__(self, cutoff_scale,
                 split_soft=False, use_veto=True,
                 **kwargs):
        super().__init__(cutoff_scale, split_soft, use_veto)

        self.beta = kwargs.get('beta', 2)
        self.alpha = kwargs.get('alpha', 1.5)

    def generation_pdf(self, momentum_fraction, theta,
                       **kwargs):
        """The probability density function for the generation
        of the scales associated with emissions."""
        # DEBUG: factor of pi missing?
        eff_coupling = self.alpha * kwargs['CR']
        return 2 * eff_coupling / (momentum_fraction * theta)

    def true_pdf(self, momentum_fraction, theta, **kwargs):
        """The true probability density function associated with
        the splitting of a parton."""
        raise NotImplementedError

    def get_scale(self, momentum_fraction, theta):
        """The angularity associated with the splitting of a parton."""
        return momentum_fraction * theta**self.beta


    def random_z_and_theta(self, scale: float) -> (float, float):
        """A method which returns a (log-uniform) random z and theta
        given an input scale (an angularity)."""
        rand  = random.random()
        z = (2.*scale)**rand / 2.
        theta = (2.*scale)**((1-rand) / self.beta)

        return z, theta

    def random_emission_scale(self, initial_scale: float,
                              **kwargs) -> float:
        """A method which returns a random emission angularity
        given the initial scale."""
        eff_coupling = self.alpha * kwargs['CR']
        eff_coupling = eff_coupling / (np.pi * self.beta)
        rand = random.random()
        final_scale = np.exp(-np.sqrt(np.log(2.*initial_scale)**2.
                                    - np.log(rand)/eff_coupling)
                          ) / 2.
        return final_scale
        # Note: there is a typo in Eqn 5.4 of
        # https://arxiv.org/pdf/1307.1699.pdf#page=18&zoom=200,0,300%5D
        # (a stray factor of 2 on the RHS),
        # but the conclusion in Equation 5.5, which we use
        # here, is correct


# =====================================
# Angularity-based shower algorithm for gauge theories
# =====================================
class GaugeTheoryShowerAlgorithm(AngularityShowerAlgorithm):
    def __init__(self, cutoff_scale,
                 split_soft=False, use_veto=True,
                 **kwargs):
        super().__init__(cutoff_scale, split_soft, use_veto, **kwargs)

        # - - - - - - - - - - - - - - - - -
        # Coupling setup
        # - - - - - - - - - - - - - - - - -
        coupling = kwargs.get('coupling', None)
        if coupling is None:
            try:
                alpha0 = kwargs['alpha0']
                mu0 = kwargs['mu0']
                NF = kwargs['NF']
                NC = kwargs['NC']
                masses = kwargs.get('masses', None)
                mu_freeze = kwargs.get('mu_freeze', None)
            except KeyError as exc:
                raise KeyError("Must specify either `coupling` or "
                               "parameters which specify a gauge theory "
                               "coupling (alpha0, mu0, NF, NC)") from exc
            def coupling(mu, **kwargs):
                return alpha_gauge_1loop(alpha0, mu0, mu, NC, NF,
                                         masses, mu_freeze)

        self.coupling = coupling

        # - - - - - - - - - - - - - - - - -
        # Splitting function setup
        # - - - - - - - - - - - - - - - - -
        splitting_function = kwargs.get('splitting_function',
                                        None)
        if splitting_function is None:
            try:
                NF = kwargs['NF']
                NC = kwargs['NC']
                accuracy = kwargs['accuracy']
            except KeyError as exc:
                raise KeyError("Must specify either `splitting_function` or "
                               "parameters which specify a splitting function "
                               "(NF, NC, accuracy)") from exc
            splitting_function = GaugeTheorySplittingFunction(**kwargs)

        self.splitting_function = splitting_function


    def true_pdf(self, momentum_fraction, theta, **kwargs):
        """The true probability density function associated with
        the splitting of a parton."""
        eff_coupling = self.coupling(self.get_scale(momentum_fraction, theta),
                                    **kwargs)
        eff_coupling *= kwargs['CR']

        splitting_fn = self.splitting_function(momentum_fraction, **kwargs)

        # DEBUG: Not sure if I want to use p(z)/theta this way -- a bit of a lie
        return eff_coupling * splittting_fn / theta


def qcd_shower_algorithm(cutoff_scale: float = 1e-8,
                         alpha0: float = ALPHA_S_ZMASS,
                         mu0: float = Z_MASS,
                         NF: int = 5,
                         NC: int = 3,
                         accuracy: str = 'MLL',
                         masses=quark_masses,
                         mu_freeze=0,
                         **kwargs
                         ) -> ShowerAlgorithm:
    """
        particle_type: str
        accuracy: str
        NF: int
        NC: int
        radius: float
    """
    return GaugeTheoryShowerAlgorithm(cutoff_scale=cutoff_scale,
                                      alpha0=alpha0, mu0=mu0,
                                      NF=NF, NC=NC,
                                      accuracy=accuracy,
                                      masses=masses,
                                      mu_freeze=mu_freeze,
                                      **kwargs)
