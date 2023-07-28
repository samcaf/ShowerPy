import pytest
from showerpy.parton import Parton
from showerpy.qcd_utils import CF, CA

def test_Parton_init():
    momentum = [100, 50, 30]
    mass = 5
    mother = Parton([50, 20, 10])
    daughters = [Parton([60, 25, 15]), Parton([40, 15, 5])]
    pid = 123
    is_final_state = False
    particle_type = 'fermion'
    NC = 3

    parton = Parton(momentum, mass=mass, mother=mother,
                    daughters=daughters, pid=pid,
                    is_final_state=is_final_state,
                    particle_type=particle_type, NC=NC)

    assert parton.momentum.as_list() == momentum
    assert parton.mass == mass
    assert parton.mother == mother
    assert parton.daughters == daughters
    assert parton.pid == pid
    assert parton.is_final_state == is_final_state
    assert parton.metadata['particle_type'] == particle_type.lower()
    assert parton.metadata['CR'] == CF(NC)


def test_Parton_split():
    momentum = [100, 50, 30]
    parton = Parton(momentum)

    momentum_fraction = 0.7
    theta = 0.3
    parton.split(momentum_fraction, theta)

    daughters = parton.daughters
    assert len(daughters) == 2
    assert daughters[0].momentum.mag() > daughters[1].momentum.mag()


def test_Parton_daughter_tree():
    momentum = [100, 50, 30]
    parton = Parton(momentum)

    momentum_fraction = 0.7
    theta = 0.3
    parton.split(momentum_fraction, theta)

    daughter_tree = parton.daughter_tree()
    assert len(daughter_tree) == 3
    assert parton in daughter_tree
    assert parton.daughters[0] in daughter_tree
    assert parton.daughters[1] in daughter_tree


# Run the tests
if __name__ == '__main__':
    pytest.main()
