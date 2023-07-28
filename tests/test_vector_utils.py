import pytest
from showerpy.vector_utils import Vector, FourVector, contract, rand_perp_vector, angle

def test_Vector_init():
    vector = Vector([1, 2, 3])
    assert vector.vector.tolist() == [1, 2, 3]

def test_Vector_dim():
    vector = Vector([1, 2, 3])
    assert vector.dim() == 3

def test_Vector_mag2():
    vector = Vector([1, 2, 3])
    assert vector.mag2() == 14

def test_Vector_mag():
    vector = Vector([1, 2, 3])
    assert vector.mag() == pytest.approx(3.7416573867739413)

def test_Vector_perp2():
    vector = Vector([3, 4, 5])
    assert vector.perp2() == 25

def test_Vector_perp():
    vector = Vector([3, 4, 5])
    assert vector.perp() == 5

def test_Vector_theta():
    vector = Vector([1, 2, 3, 4])
    assert vector.theta == pytest.approx(0.75204008, 1e-6)

def test_Vector_phi():
    vector = Vector([1, 2, 3, 4])
    assert vector.phi == pytest.approx(1.10714872)

def test_Vector_unit():
    vector = Vector([1, 2, 3])
    unit_vector = vector.unit()
    assert unit_vector.vector.tolist() == [0.2672612419124244, 0.5345224838248488, 0.8017837257372732]

def test_Vector_contract():
    vector1 = Vector([1, 2, 3])
    vector2 = Vector([4, 5, 6])
    assert vector1.contract(vector2) == 32

def test_Vector_angle():
    vector1 = Vector([1, 0, 0])
    vector2 = Vector([0, 1, 0])
    assert vector1.angle(vector2) == pytest.approx(1.5707963267948966)

def test_Vector_rotate_around():
    vector = Vector([1, 0, 0])
    axis = Vector([0, 0, 1])
    rotated_vector = vector.rotate_around(axis, 0.5)
    assert rotated_vector.vector.tolist() == pytest.approx([0.8775825618903728, 0.479425538604203, 0])

def test_Vector_rand_perp_vector():
    vector = Vector([1, 2, 3])
    perp_vector = vector.rand_perp_vector()
    assert vector.contract(perp_vector) == pytest.approx(0)

def test_FourVector_init():
    four_vector = FourVector([1, 2, 3, 4])
    assert four_vector.vector.tolist() == [1, 2, 3, 4]

def test_FourVector_eta():
    four_vector = FourVector([1, 2, 3, 4])
    assert four_vector.eta == pytest.approx(0.9566555518497877)

def test_FourVector_y():
    four_vector = FourVector([5.47722557505, 2, 3, 4])
    assert four_vector.y == pytest.approx(0.929362947766225023)

def test_FourVector_m2():
    four_vector = FourVector([1, 2, 3, 4])
    assert four_vector.m2() == -28

def test_FourVector_m():
    four_vector = FourVector([5.47722557505, 2, 3, 4])
    assert four_vector.m() == pytest.approx(1)

def test_FourVector_mag2():
    four_vector = FourVector([1, 2, 3, 4])
    assert four_vector.mag2() == 29

# Add more test cases for the remaining methods in the Vector and FourVector classes


# Run the tests
if __name__ == '__main__':
    pytest.main()
