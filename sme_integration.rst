.. Discussion of stochastic integration considerations

SME integration
===============

In order to integrate a stochastic equation:

.. math::

   dX=a(X,t)dt+b(X,t)dW

if one wants to be more sophisticated than Euler integration, one can use
Milstein integration:

.. math::

   X_{i+1}=X_i+a(X_i,t_i)\Delta t_i+b(X_i,t_i)\Delta W_i+
   \frac{1}{2}b(X_i,t_i)\frac{\partial}{\partial X}b(X_i,t_i)\left(
   (\Delta W_i)^2-\Delta t_i\right)

What if we are interested in a vector-valued equation:

.. math::

   d\vec{\rho}=\vec{a}(\vec{\rho},t)dt+\vec{b}(\vec{\rho},t)dW

The way to generalize the Milstein scheme (while still restricting ourselves to
a scalar-valued Wiener process) is:

.. math::

   \vec{\rho}_{i+1}=\vec{\rho}_i+\vec{a}(\vec{\rho}_i,t_i)\Delta t_i+
   \vec{b}(\vec{\rho}_i,t_i)\Delta W_i+
   \frac{1}{2}\left(\vec{b}(\vec{\rho}_i,t_i)\cdot\vec{\nabla}_{\vec{\rho}}
   \right)\vec{b}(\vec{\rho}_i,t_i)\left((\Delta W_i)^2-\Delta t_i\right)

For
:math:`\vec{b}(\vec{\rho},t)=(G+\vec{k}\cdot\vec{\rho})\vec{\rho}` as defined in
:doc:`vectorizations` we can write:

.. math::

   \left(\vec{b}(\vec{\rho},t)\cdot\vec{\nabla}_{\vec{\rho}}\right)
   \vec{b}(\vec{\rho}_i,t_i)=\left((\vec{k}^TG\vec{\rho})+G^2+
   2(\vec{k}\cdot\vec{\rho})\left(G+
   \vec{k}\cdot\vec{\rho}\right)\right)\vec{\rho}
