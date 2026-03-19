import warnings


def isometry(epsilon, origin):
    """
    An isometry of the ellipticity *epsilon* that maps *origin* to the
    origin.
    """
    xp = epsilon.__array_namespace__()

    return (epsilon - origin) / (1 - xp.conj(origin) * epsilon)


def normal_coordinates(epsilon, origin=None):
    """
    Compute normal coordinates of ellipticity *epsilon* in *origin*
    (default 0).
    """
    xp = epsilon.__array_namespace__()

    if origin is not None:
        epsilon = isometry(epsilon, origin)

    r = xp.abs(epsilon)
    u = epsilon / xp.where(r > 0, r, 1.0)

    return 2 * xp.atanh(r) * u


def exponential_map(vec, origin=None):
    """
    Compute the exponential map of vector *vec* in origin *origin*
    (default 0).
    """
    xp = vec.__array_namespace__()

    r = xp.abs(vec)
    u = vec / xp.where(r > 0, r, 1.0)

    epsilon = xp.tanh(r / 2) * u

    if origin is not None:
        epsilon = isometry(epsilon, -origin)

    return epsilon


def distance(x, y=None):
    """
    Compute the intrinsic distance between ellipticities.
    """
    xp = x.__array_namespace__()

    if y is None:
        return 2 * xp.atanh(xp.abs(x))

    p = xp.abs(x)
    q = xp.abs(y)
    r = xp.abs(x - y)

    return xp.acosh(1 + 2 * r**2 / (((1 - p) * (1 + p)) * ((1 - q) * (1 + q))))


def mean(epsilon, weight=None, *, axis=None, initial=None, maxiter=100, tol=1e-6):
    """
    Compute the Fréchet mean of a sample of ellipticities *epsilon*.  If
    *weight* is given, the mean is weighted.
    """
    xp = epsilon.__array_namespace__()

    if weight is None:
        weight = 1.0

    wtot = xp.sum(0 * epsilon + weight, axis=axis)

    if initial is None:
        initial = xp.sum(weight * epsilon, axis=axis) / wtot

    mu = initial

    # gradient descent rate
    a = 0.5

    for _ in range(maxiter):
        n = normal_coordinates(epsilon, mu)
        v = xp.sum(weight * n, axis=axis) / wtot
        if xp.all(xp.abs(v) < tol):
            break
        mu = exponential_map(a * v, mu)
    else:
        warnings.warn("mean did not converge")

    return mu
