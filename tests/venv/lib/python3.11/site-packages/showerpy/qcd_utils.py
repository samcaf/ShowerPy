import numpy as np

# Physical parameters
Z_MASS = 91.19 # GeV
quark_masses = [0.0023, 0.0048, 0.095, 1.275, 4.18]
quark_masses = np.array(quark_masses)


# =====================================
# Gauge Theory Basics
# =====================================
# ---------------------------------------------------
# Group theory factors and other constants
# ---------------------------------------------------
# Dynkin index of the fundamental representation
TF = 1./2.

def CF(NC):
    """Returns the Casimir of the fundamental representation
    of SU(NC)"""
    return (NC**2. - 1.)/(2.*NC)

def CA(NC):
    """Returns the Casimir of the adjoint representation
    of SU(NC)"""
    return NC


# Number of active flavors
def n_active_flavors(mu, masses=quark_masses):
    """Returns the number of active flavors at the scale mu
    with masses masses.

    Gives the value for QCD by default
    """
    return np.sum(masses < mu)


# ---------------------------------------------------
# Couplings
# ---------------------------------------------------
# - - - - - - - - - - - - - - - - -
# Beta function
# - - - - - - - - - - - - - - - - -
# Recall the definition of the beta_i:
#   beta(g) = mu d/dmu g = - beta_0/4pi g^3 - O(g^5)
# or
#   beta(alpha) = mu d/dmu alpha = -2 beta_0 alpha^2 + O(alpha^3)

# Leading beta function coefficient with NC colors,
# NF flavors of (dirac) fermions (or scalars)
def beta0(NC, NF):
    return (11/3 * NC - 2/3 * NF)/(4*np.pi)


# - - - - - - - - - - - - - - - - -
# Coupling constants
# - - - - - - - - - - - - - - - - -
# Generic coupling
def alpha_gauge_1loop(alpha0, mu0, mu, NC, NF,
                      masses=None, mu_freeze=None):
    """1-loop running coupling constant for a SU(NC) gauge theory with
    NF flavors of (dirac) fermions, and NS flavors of scalars
    """
    if mu_freeze is not None:
        mu = np.maximum(mu, mu_freeze)

    if masses is None:
        return alpha0/(1 + 2*alpha0*beta0(NC, NF)*np.log(mu/mu0))
    # If there are masses, we need to integrate them out sequentially.
    # We do this recursively.
    # First, make sure we have a mass for every flavor and find the
    # largest mass
    assert len(masses) == NF, \
            "Need a mass for every flavor."
    largest_mass = np.max(masses)

    # If we are above the mass threshold, no change is needed
    if mu >= largest_mass:
        return alpha_gauge_1loop(alpha0, mu0, mu, NC, NF)
    # Otherwise, we need to integrate out the heaviest flavor
    else:
        # Find the index of the heaviest flavor
        heaviest_flavor = np.argmax(masses)
        # Remove the heaviest flavor from the list
        masses = np.delete(masses, heaviest_flavor)
        # Reduce the number of flavors by one
        NF -= 1
        # Find the coupling at the scale of the heaviest flavor
        alpha_heaviest = alpha_gauge_1loop(alpha0, mu0, largest_mass,
                                           NC, NF, masses)
        # Recursively call the function
        return alpha_gauge_1loop(alpha0=alpha_heaviest, mu0=largest_mass,
                                 mu=mu, NC=NC, NF=NF, masses=masses)


# QED coupling
ALPHA_EM_ZMASS = 1./127

def alpha_em(mu):
    """1-loop coupling for QED with an electron. Argument in GeV."""
    return alpha_gauge_1loop(ALPHA_EM_ZMASS, Z_MASS, mu, NC=0, NF=1)


# QCD coupling
ALPHA_S_ZMASS = 0.118
LAMBDA_QCD = 0.3 # GeV, approximate

def alpha_s(mu, masses=quark_masses,
            mu_freeze=LAMBDA_QCD):
    """Returns the QCD coupling constant at the scale mu
    with 5 flavors of quarks
    """
    return alpha_gauge_1loop(ALPHA_S_ZMASS, Z_MASS, mu, NC=3, NF=5,
                             masses=masses, mu_freeze=mu_freeze)



# ---------------------------------------------------
# Splitting functions
# ---------------------------------------------------
class SplittingFunction:
    """Base class for splitting functions"""
    def __init__(self, **kwargs):
        reduced = kwargs.get('reduced', False)

        self._dict = {'reduced': reduced}

    def __call__(self, z, **kwargs):
        if self._dict['reduced']:
            assert (0 < z < 1/2), "z must be in (0, 1/2)"
            return self._splitting_function(z, **kwargs) \
                    + self._splitting_function(1-z, **kwargs)
        else:
            assert (0 < z < 1), "z must be in (0, 1)"
            return self._splitting_function(z, **kwargs)


    def _splitting_function(self, z, **kwargs):
        raise NotImplementedError("Splitting function not implemented")


# - - - - - - - - - - - - - - - - -
# Splitting functions in gauge theory
# - - - - - - - - - - - - - - - - -
def F2FG_splitting_function(z, NC, accuracy='MLL', **kwargs):
    """Splitting function for a fermion with NC colors"""
    if accuracy == 'LL':
        return 2*CF(NC)/z
    return 2 * CF(NC) * ((1 + (1 - z)**2.)/z)

def G2GG_splitting_function(z, NC, NF, accuracy='MLL', **kwargs):
    """Splitting function for a gauge boson with NC colors"""
    if accuracy == 'LL':
        return 2*CA(NC)/z
    return CA(NC) * (
        2.*(1.-z)/z + z*(1.-z)
        + (TF*NF)*(z**2.+(1.-z)**2.)/CA(NC)
        )


class GaugeTheorySplittingFunction(SplittingFunction):
    """Splitting function in a gauge theory (QCD by default)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dict['NC'] = kwargs.get('NC', 3)
        self._dict['NF'] = kwargs.get('NF', 5)

        accuracy = kwargs.get('accuracy', 'MLL')
        if accuracy in ['singular', 'LL']:
            accuracy = 'LL'
        elif accuracy in ['hard-collinear', 'MLL']:
            accuracy = 'MLL'
        else:
            raise ValueError(f"Unrecognized accuracy: {accuracy}")
        self._dict['accuracy'] = accuracy

        self.default_split_type = kwargs.get('default_split_type',
                                             'F2FG')

    def _splitting_function(self, z, **kwargs):
        split_type = kwargs.get('split_type', self.default_split_type)
        split_type = split_type.upper()
        if split_type == 'F2FG':
            return F2FG_splitting_function(z, **self._dict)
        elif split_type == 'G2GG':
            return G2GG_splitting_function(z, **self._dict)
