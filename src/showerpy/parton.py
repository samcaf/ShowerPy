import math as m
from showerpy.qcd_utils import CF, CA
from showerpy.vector_utils import Vector, angle

# =====================================
# Parton class
# =====================================
class Parton():
    """A class which encodes the information contained by a parton,
    including momentum and daughter partons due to splittings.
    Also includes methods to allow partons to split.
    """
    def __init__(self, momentum, **kwargs):
        # Initializing kinematic information
        if isinstance(momentum, Vector):
            self.momentum = momentum
        else:
            self.momentum = Vector(momentum)

        self.mass = kwargs.pop('mass', 0)
        self.energy = m.sqrt(self.momentum.mag2() + self.mass**2.)

        # Initializing information for branching
        self.mother = kwargs.pop('mother', None)
        self.daughters = kwargs.pop('daughters', [])

        # Initializing extra identifiers
        self.pid = kwargs.pop('pid', None)
        self.is_final_state = kwargs.pop('is_final_state', True)

        # Additional identifiers for more complicated manipulations
        self.metadata = {}

        # Allowing several options for particle types
        particle_type = kwargs.pop('particle_type', None)
        color_rep = kwargs.pop('color_rep', None)

        # Setting up particle type in case its relevant for splitting
        if particle_type is not None:
            if particle_type.lower() in ['fermion', 'f', 'quark', 'q']:
                self.metadata['particle_type'] = 'fermion'
            elif particle_type.lower() in ['gauge', 'gauge boson',
                                           'gluon', 'g']:
                assert color_rep in ['adjoint', 'a', None], \
                    "Gauge bosons must be in the adjoint representation."
                color_rep = 'adjoint'
                self.metadata['particle_type'] = 'gauge boson'
        else:
            self.metadata['particle_type'] = None

        # Storing the color representation after processing
        self.metadata['color_rep'] = color_rep

        self.metadata.update(kwargs)


    def split(self, momentum_fraction, theta,
              emission_metadata=None,
              remainder_metadata=None):
        """A method which splits a parton into two
        daughter partons separated by angle theta,
        of which the softer parton has momentum fraction:
        |p_soft| = energy_fraction |p_original|

        The daughter partons are then encoded into this
        parton as self.daughters.
        """
        k_em, k_rem = self.momentum.split(momentum_fraction, theta)

        particle_type = self.metadata['particle_type']
        if particle_type == 'fermion':
            self.metadata['split_type'] = 'F2FG'
            if remainder_metadata is None:
                remainder_metadata = {'particle_type': 'fermion',
                                      'color_rep': self.metadata['color_rep']}
            if emission_metadata is None:
                emission_metadata = {'particle_type': 'gauge boson',
                                     'color_rep': 'adjoint'}
        elif particle_type == 'gauge boson':
            self.metadata['split_type'] = 'G2GG'
            if remainder_metadata is None:
                remainder_metadata = {'particle_type': 'gauge boson',
                                      'color_rep': self.metadata['color_rep']}
            if emission_metadata is None:
                emission_metadata = {'particle_type': 'gauge boson',
                                     'color_rep': 'adjoint'}
        else:
            if remainder_metadata is None:
                remainder_metadata = {}
            if emission_metadata is None:
                emission_metadata = {}

        emission = Parton(k_em, **emission_metadata)
        remainder = Parton(k_rem, **remainder_metadata)

        # Setting daughter1 to be the harder of the two
        if k_em.mag() > k_rem.mag():
            daughter1, daughter2 = emission, remainder
        else:
            daughter1, daughter2 = remainder, emission

        # Setting the mother and daughters of the partons
        daughter1.mother, daughter2.mother = self, self
        self.daughters = [daughter1, daughter2]
        self.is_final_state = False

        assert daughter1.momentum.mag() >= daughter2.momentum.mag(),\
            "Daughter 1 momentum must be harder than or equal to "\
            "daughter 2 momentum."


    def angle(self, other):
        """A method which returns the angle between this parton
        and another parton.
        """
        return angle(self.momentum, other.momentum)


    def __add__(self, other):
        """A method which adds the momentum of this parton
        to another parton.
        """
        return Parton(self.momentum + other.momentum)


    def daughter_tree(self):
        """Returns a list of all daughter partons
        (i.e. the jet associated with the parton).
        """
        jet = [self]
        for daughter in self.daughters:
            jet += daughter.daughter_tree()
        return jet
