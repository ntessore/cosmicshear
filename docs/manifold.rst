The ellipticity manifold
========================

The space of ellipticities is the hyperbolic plane, which is a Riemannian
manifold.  More specifically, :math:`\epsilon`-ellipticities are points in the
Poincaré disk model of the hyperbolic plane.

Distance
--------

.. autofunction:: cosmicshear.distance


Isometry
--------

The hyperbolic plane has an isometry :math:`T_{\epsilon_0}` that maps the
point :math:`\epsilon_0` to the origin,

.. math::

    T_{\epsilon_0}(\epsilon)
    = \frac{\epsilon - \epsilon_0}{1 - \epsilon_0^* \epsilon} \;.

In weak lensing, the action of a reduced shear :math:`g` is precisely
:math:`T_{-g}`.


.. autofunction:: cosmicshear.isometry


Exponential map
---------------

.. autofunction:: cosmicshear.exponential_map
.. autofunction:: cosmicshear.normal_coordinates


Intrinsic mean
--------------

.. autofunction:: cosmicshear.mean
