def chi_from_epsilon(eps):
    r"""
    Transform epsilon-ellipticity to chi-ellipticity.

    .. math::

        \chi = \frac{2\epsilon}{1 + |\epsilon|^2}

    """
    return (2 * eps) / (1 + eps.real**2 + eps.imag**2)


def epsilon_from_chi(chi):
    r"""
    Transform chi-ellipticity to epsilon-ellipticity.

    .. math::

        \epsilon = \frac{\chi}{1 + \sqrt{1 - |\chi|^2}}

    """
    xp = chi.__array_namespace__()

    return chi / (1 + xp.sqrt(1 - (chi.real**2 + chi.imag**2)))


def transform(jac, eps=None):
    r"""
    Transform epsilon-ellipticity between coordinate systems.

    Uses the Jacobian matrix *jac* (or a stack of matrices).  Transforms
    the ellipticity *eps* if given, or zero ellipticity otherwise.

    Internally, the code computes the shear transformation via the
    decomposition

    .. math::

        \mathrm{J} \begin{pmatrix}
            1 + \chi_1 & \chi_2 \\
            \chi_2 & 1 - \chi_1
        \end{pmatrix} \mathrm{J}^{\mathsf{T}} \\
        = \frac{1}{1 + |\epsilon|^2} \left[
            \mathrm{J} \begin{pmatrix}
                1 + \epsilon_1 & \epsilon_2 \\
                \epsilon_2 & 1 - \epsilon_1
            \end{pmatrix}
        \right]
        \left[
            \mathrm{J} \begin{pmatrix}
                1 + \epsilon_1 & \epsilon_2 \\
                \epsilon_2 & 1 - \epsilon_1
            \end{pmatrix}
        \right]^{\mathsf{T}} \;.

    """
    xp = jac.__array_namespace__()

    if eps is not None:
        a = xp.stack([eps.real, eps.imag, eps.imag, -eps.real], axis=-1)
        jac = jac + (jac @ a.reshape(a.shape[:-1] + (2, 2)))
        del a

    x = (jac[..., 0, 0] + jac[..., 1, 1]) + 1j * (jac[..., 1, 0] - jac[..., 0, 1])
    y = (jac[..., 0, 0] - jac[..., 1, 1]) + 1j * (jac[..., 1, 0] + jac[..., 0, 1])

    # this is y/x.conj() or x/y.conj() without having to deal with the zeros
    return (x * y) / xp.maximum(x.real**2 + x.imag**2, y.real**2 + y.imag**2)


def inverse_transform(jac, eps=None):
    r"""
    Inverse-transform epsilon-ellipticity between coordinate systems.

    Uses the Jacobian matrix *jac* (or a stack of matrices).  Transforms
    the ellipticity *eps* if given, or zero ellipticity otherwise.

    Equivalent to ``transform(inv(jac), eps)``, but does not compute the
    inverse explicitly.

    Internally, the code computes the shear transformation via the
    decomposition

    .. math::

        \mathrm{J}^{-1} \begin{pmatrix}
            1 + \chi_1 & \chi_2 \\
            \chi_2 & 1 - \chi_1
        \end{pmatrix} \mathrm{J}^{-\mathsf{T}} \\
        = \frac{(1 - |\epsilon|^2)^2}{1 + |\epsilon|^2} \left[
            \begin{pmatrix}
                1 - \epsilon_1 & -\epsilon_2 \\
                -\epsilon_2 & 1 + \epsilon_1
            \end{pmatrix} \mathrm{J}
        \right]^{-1}
        \left[
            \begin{pmatrix}
                1 - \epsilon_1 & -\epsilon_2 \\
                -\epsilon_2 & 1 + \epsilon_1
            \end{pmatrix} \mathrm{J}
        \right]^{-\mathsf{T}} \;.

    """
    xp = jac.__array_namespace__()

    if eps is not None:
        a = xp.stack([-eps.real, -eps.imag, -eps.imag, eps.real], axis=-1)
        jac = jac + (a.reshape(a.shape[:-1] + (2, 2)) @ jac)
        del a

    x = (jac[..., 1, 1] + jac[..., 0, 0]) - 1j * (jac[..., 1, 0] - jac[..., 0, 1])
    y = (jac[..., 1, 1] - jac[..., 0, 0]) - 1j * (jac[..., 1, 0] + jac[..., 0, 1])

    # this is y/x.conj() or x/y.conj() without having to deal with the zeros
    return (x * y) / xp.maximum(x.real**2 + x.imag**2, y.real**2 + y.imag**2)
