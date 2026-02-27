Ellipticities
=============

Definitions
-----------

epsilon-ellipticity
^^^^^^^^^^^^^^^^^^^

.. math::

    \epsilon = \frac{1 - q}{1 + q} \, e^{2i\phi}


chi-ellipticity
^^^^^^^^^^^^^^^

.. math::

    \chi = \frac{1 - q^2}{1 + q^2} \, e^{2i\phi}


Converting between definitions
------------------------------

.. autofunction:: cosmicshear.chi_from_epsilon
.. autofunction:: cosmicshear.epsilon_from_chi


Coordinate transformations
--------------------------

.. autofunction:: cosmicshear.transform
.. autofunction:: cosmicshear.inverse_transform
