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
   \vec{b}(\vec{\rho},t)=\left(\left(\vec{k}^TG\vec{\rho}\right)+G^2+
   2(\vec{k}\cdot\vec{\rho})\left(G+
   \vec{k}\cdot\vec{\rho}\right)\right)\vec{\rho}

Order 1.5 Taylor scheme
-----------------------

To get higher order convergence in time, we can use a more complicated update
formula (restricting ourselves to :math:`\vec{a}` and :math:`\vec{b}` with no
explicit time dependence, as we have in our problem):

.. math::

   \begin{align}
   \vec{\rho}_{i+1}&=\vec{\rho}_i+\vec{a}(\vec{\rho}_i)\Delta t_i+
   \vec{b}(\vec{\rho}_i)\Delta W_i+
   \frac{1}{2}\left(\vec{b}(\vec{\rho}_i)\cdot\vec{\nabla}_{\vec{\rho}}
   \right)\vec{b}(\vec{\rho}_i)\left((\Delta W_i)^2-\Delta t_i\right)+ \\
   &\quad\left(\vec{b}(\vec{\rho}_i)\cdot\vec{\nabla}_{\vec{\rho}}
   \right)\vec{a}(\vec{\rho}_i)\Delta Z_i+\left(\vec{a}(\vec{\rho}_i)\cdot
   \vec{\nabla}_{\vec{\rho}}+\frac{1}{2}\vec{b}^\mathsf{T}(\vec{\rho})D^2
   \vec{b}(\vec{\rho})\right)
   \vec{b}(\vec{\rho}_i)\left(
   \Delta W_i\Delta t_i-\Delta Z_i\right)+ \\
   &\quad\frac{1}{2}\left(\vec{a}(\vec{\rho}_i)\cdot\vec{\nabla}_{\vec{\rho}}
   +\frac{1}{2}\vec{b}^\mathsf{T}(\vec{\rho})D^2
   \vec{b}(\vec{\rho})\right)\vec{a}(\vec{\rho}_i)\Delta t_i^2+
   \frac{1}{2}\left(
   \vec{b}(\vec{\rho}_i)\cdot\vec{\nabla}_{\vec{\rho}}
   \right)^2\,\vec{b}(\vec{\rho}_i)\left(\frac{1}{3}(\Delta W_i)^2-
   \Delta t_i\right)\Delta W_i
   \end{align}

Where :math:`\vec{b}^\mathsf{T}(\vec{\rho})D^2\vec{b}(\vec{\rho})`
is horrible shorthand for
:math:`\sum_{j,k}b^jb^k\frac{\partial^2}{\partial x^j\partial x^k}`. These terms are currently missing from the python implementation.

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
   \left(\vec{b}(\vec{\rho})\cdot\vec{\nabla}_{\vec{\rho}}\right)\vec{a}(
   \vec{\rho})&=QG\vec{\rho}+(\vec{k}\cdot\vec{\rho})Q\vec{\rho} \\
   \left(\vec{a}(\vec{\rho})\cdot\vec{\nabla}_{\vec{\rho}}\right)\vec{b}(
   \vec{\rho})&=GQ\vec{\rho}+(\vec{k}\cdot\vec{\rho})Q\vec{\rho}+\left(
   \vec{k}^TQ\vec{\rho}\right)\vec{\rho} \\
   \left(\vec{a}(\vec{\rho})\cdot\vec{\nabla}_{\vec{\rho}}\right)\vec{a}(
   \vec{\rho})&=Q^2\vec{\rho} \\
   \left(\vec{b}(\vec{\rho})\cdot\vec{\nabla}_{\vec{\rho}}\right)^2\,\vec{b}(
   \vec{\rho})&=G^3\vec{\rho}+3(\vec{k}\cdot\vec{\rho})G^2\vec{\rho}+
   3\left(\vec{k}^TG\vec{\rho}+
   2(\vec{k}\cdot\vec{\rho})^2\right)G\vec{\rho}+ \\
   &\quad\left(\vec{k}^TG^2\vec{\rho}+6(\vec{k}\cdot\vec{\rho})
   \vec{k}^TG\vec{\rho}+6(\vec{k}\cdot\vec{\rho})^3\right)\vec{\rho}
   \end{align}

We explore testing the convergence rates in :doc:`testing`.
