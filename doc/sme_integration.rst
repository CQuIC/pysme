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

Vector Milstein
---------------

What if we are interested in a vector-valued equation:

.. math::

   d\vec{\rho}=\vec{a}(\vec{\rho},t)dt+\vec{b}(\vec{\rho},t)dW

The way to generalize the Milstein scheme (while still restricting ourselves to
a scalar-valued Wiener process) is

.. math::

   \rho^\mu_{i+1}=\rho^\mu_i+a^\mu_i\Delta t_i+b^\mu_i\Delta W_i+
   \frac{1}{2}b^\nu_i\partial_\nu b^\mu_i\left((\Delta W_i)^2
   -\Delta t_i\right),

where I have adopted an index convention for vectors such that

.. math::

   \begin{align}
   \vec{\rho}&=\rho^\mu\hat{e}_\mu \\
   a^\mu_i&=a^\mu(\vec{\rho}_i,t_i) \\
   \partial_\nu&=\frac{\partial}{\partial\rho^\nu},
   \end{align}

and indices that appear in both upper and lower positions in the same term are
implicitly summer over.

For
:math:`b^\mu=G^\mu_\nu\rho^\nu+k_\nu\rho^\nu\rho^\mu` as defined in
:doc:`vectorizations` we can write:

.. math::

   \begin{align}
   b^\nu\partial_\nu b^\mu&=\left(k_\nu G^\nu_\sigma\rho^\mu+
   G^\mu_\nu G^\nu_\sigma+2k_\nu\rho^\nu(G^\mu_\sigma
   +k_\sigma\rho^\mu)\right)\rho^\sigma \\
   b^\nu\partial_\nu b^\mu\hat{e}_\mu&=\left(
   \left(\vec{k}^\mathsf{T}G\vec{\rho}\right)
   +G^2+2(\vec{k}\cdot\vec{\rho})\left(G+\vec{k}\cdot
   \vec{\rho}\right)\right)\vec{\rho}
   \end{align}

Order 1.5 Taylor scheme
-----------------------

To get higher order convergence in time, we can use a more complicated update
formula (restricting ourselves to :math:`\vec{a}` and :math:`\vec{b}` with no
explicit time dependence, as we have in our problem):

.. math::

   \begin{align}
   \rho^\mu_{i+1}&=\rho^\mu_i+a^\mu_i\Delta t_i+
   b^\mu_i\Delta W_i+\frac{1}{2}b^\nu_i\partial_\nu b^\mu_i\left(
   (\Delta W_i)^2-\Delta t_i\right)+ \\
   &\quad b^\nu_i\partial_\nu a^\mu_i\Delta Z_i
   +\left(a^\nu_i\partial_\nu
   +\frac{1}{2}b^\nu_ib^\sigma_i\partial_\nu\partial_\sigma\right)
   b^\mu_i\left(\Delta W_i\Delta t_i-\Delta Z_i\right)+ \\
   &\quad\frac{1}{2}\left(a^\nu_i\partial_\nu
   +\frac{1}{2}b^\nu_ib^\sigma_i\partial_\nu\partial_\sigma\right)
   a^\mu_i\Delta t_i^2
   +\frac{1}{2}b^\nu_i\partial_\nu b^\sigma_i\partial_\sigma b^\mu_i\left(
   \frac{1}{3}(\Delta W_i)^2-\Delta t_i\right)\Delta W_i
   \end{align}

Recall from :doc:`vectorizations` that:

.. math::

   \begin{align}
   \vec{a}(\vec{\rho})&=Q\vec{\rho} \\
   Q&:=(N+1)D(\vec{c})+ND(\vec{c}^*)+E(M,\vec{c})+F(\vec{h})
   \end{align}

:math:`\Delta Z` is a new random variavle related to :math:`\Delta W`:

.. math::

   \begin{align}
   \Delta W_i&=U_{1,i}\sqrt{\Delta t_i} \\
   \Delta Z_i&=\frac{1}{2}\left(U_{1,i}+\frac{1}{\sqrt{3}}U_{2,i}\right)
   \Delta t_i^{3/2}
   \end{align}

where :math:`U_1`, :math:`U_2` are normally distributed random variables with
mean 0 and variance 1.

The new terms in the higher-order update formula are given below:

.. math::

   \begin{align}
   b^\nu\partial_\nu a^\mu\hat{e}_\mu&=QG\vec{\rho}
   +(\vec{k}\cdot\vec{\rho})Q\vec{\rho} \\
   a^\nu\partial_\nu b^\mu\hat{e}_\mu&=GQ\vec{\rho}+
   (\vec{k}\cdot\vec{\rho})Q\vec{\rho}+\left(
   \vec{k}^\mathsf{T}Q\vec{\rho}\right)\vec{\rho} \\
   a^\nu\partial_\nu a^\mu\hat{e}_\mu&=Q^2\vec{\rho} \\
   b^\nu\partial_\nu b^\sigma\partial_\sigma b^\mu\hat{e}_\mu&=G^3\vec{\rho}
   +3(\vec{k}\cdot\vec{\rho})G^2\vec{\rho}+
   3\left(\vec{k}^\mathsf{T}G\vec{\rho}+
   2(\vec{k}\cdot\vec{\rho})^2\right)G\vec{\rho}+ \\
   &\quad\left(\vec{k}^\mathsf{T}G^2\vec{\rho}+6(\vec{k}\cdot\vec{\rho})
   \vec{k}^\mathsf{T}G\vec{\rho}
   +6(\vec{k}\cdot\vec{\rho})^3\right)\vec{\rho} \\
   b^\nu b^\sigma\partial_\nu\partial_\sigma b^\mu\hat{e}_\mu&=2\left(
   \vec{k}^\mathsf{T}G\vec{\rho}+(\vec{k}\cdot\vec{\rho})^2\right)\left(
   G\vec{\rho}+(\vec{k}\cdot\vec{\rho})\vec{\rho}\right) \\
   b^\nu b^\sigma\partial_\nu\partial_\sigma a^\mu\hat{e}_\mu&=0
   \end{align}

We explore testing the convergence rates in :doc:`testing`.
