import galsim
import numpy as np
import pytest

import cosmicshear


@pytest.fixture
def random_size():
    return 1000


@pytest.fixture
def eps(random_size, rng):
    a = rng.uniform(0.0, 1.0, random_size)
    t = rng.uniform(-np.pi, np.pi, random_size)
    return a * np.exp(1j * t)


@pytest.fixture
def chi(eps):
    return eps


@pytest.fixture
def jac(random_size, rng):
    from scipy.stats import random_correlation

    eigs = rng.exponential(size=(random_size, 2))
    eigs /= eigs.sum(axis=-1, keepdims=True) / 2

    return np.vectorize(
        random_correlation.rvs,
        otypes=[float],
        excluded={"random_state"},
        signature="(2)->(2, 2)",
    )(eigs, random_state=rng)


def test_chi_from_epsilon(eps):
    chi = cosmicshear.chi_from_epsilon(eps)

    compare = [(s := galsim.Shear(g1=g.real, g2=g.imag)).e1 + 1j * s.e2 for g in eps]

    np.testing.assert_allclose(chi, compare)


def test_epsilon_from_chi(chi):
    eps = cosmicshear.epsilon_from_chi(chi)

    compare = [(s := galsim.Shear(e1=e.real, e2=e.imag)).g1 + 1j * s.g2 for e in chi]

    np.testing.assert_allclose(eps, compare)


def test_transform_zero(jac):
    eps_tfm = cosmicshear.transform(jac)

    compare = [
        galsim.JacobianWCS(*j.ravel()).shearToWorld(galsim.Shear(0j)).shear for j in jac
    ]

    np.testing.assert_allclose(eps_tfm, compare)


def test_transform(jac, eps):
    eps_tfm = cosmicshear.transform(jac, eps)

    compare = [
        galsim.JacobianWCS(*j.ravel()).shearToWorld(galsim.Shear(e)).shear
        for j, e in zip(jac, eps)
    ]

    np.testing.assert_allclose(eps_tfm, compare)


def test_transform_scalar(jac):
    eps_tfm = cosmicshear.transform(jac, 0.1)

    compare = [
        galsim.JacobianWCS(*j.ravel()).shearToWorld(galsim.Shear(0.1 + 0j)).shear
        for j in jac
    ]

    np.testing.assert_allclose(eps_tfm, compare)


def test_inverse_transform_zero(jac):
    eps_tfm = cosmicshear.inverse_transform(jac)

    compare = [
        galsim.JacobianWCS(*j.ravel()).shearToImage(galsim.Shear(0j)).shear for j in jac
    ]

    np.testing.assert_allclose(eps_tfm, compare)


def test_inverse_transform(jac, eps):
    eps_tfm = cosmicshear.inverse_transform(jac, eps)

    compare = [
        galsim.JacobianWCS(*j.ravel()).shearToImage(galsim.Shear(e)).shear
        for j, e in zip(jac, eps)
    ]

    np.testing.assert_allclose(eps_tfm, compare)


def test_transform_roundtrip_zero(jac):
    eps_round = cosmicshear.inverse_transform(jac, cosmicshear.transform(jac))
    np.testing.assert_allclose(eps_round, 0, atol=1e-6, rtol=0.0)


def test_transform_roundtrip(eps, jac):
    eps_round = cosmicshear.inverse_transform(jac, cosmicshear.transform(jac, eps))
    np.testing.assert_allclose(eps_round, eps)
