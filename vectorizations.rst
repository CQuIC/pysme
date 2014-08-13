.. Explanation of the vectorization process

Vectorization
=============

In this module, integration of ordinary and stochastic master equations is
performed on density operators parametrized by :math:`d^2` real numbers, where
:math:`d` is the dimension of the system Hilbert space. These are the components
of the density operator as a vector in a basis that is Hermitian and, excepting
the identity, traceless. Since the ordinary and stochastic master equations
under consideration are trace preserving, one could neglect the basis element
corresponding to the identity, but as the module currently stands it is included
to simplify some expressions and provide a simple test to make sure calculations
are proceeding as they ought to.

The preliminary basis having these properties that is used consists of the
generalized Gell--Mann matrices:

.. math::

   \Lambda^{jk}=\begin{cases}
   |j\rangle\langle k|+|k\rangle\langle j|, & 1\leq k<j\leq d \\ \\
   -i|j\rangle\langle k|+i|k\rangle\langle j|, & 1\leq j<k\leq d \\ \\
   \sqrt{\frac{2}{k(k+1)}}\left(\sum_{l=1}^k|l\rangle\langle l|-
   |k\rangle\langle k|\right), & 1\leq j=k<d \\ \\
   I, & j=k=d
   \end{cases}

I have toyed around with building a custom basis to make the coupling operator
sparse by applying orthogonal transformations to the normalized version of this
basis, but since that appears to have little effect I believe I will simply use
this basis for the time being. This basis as I have written it is orthogonal,
but not normalized:

.. math::

   \operatorname{Tr}[\Lambda^{jk}\Lambda^{mn}]=\delta_{jm}\delta_{kn}\big(2+
   \delta_{jd}\delta_{kd}(d-2)\big)

The density operator and coupling operator are vectorized in the following
manner:

.. math::

   \begin{align}
   \rho &=\sum_{j,k}\rho_{jk}\Lambda^{jk}, & \rho_{jk} &\in\mathbb{R} \\
   c &=\sum_{j,k}c_{jk}\Lambda^{jk}, & c_{jk} &\in\mathbb{C}
   \end{align}

Matrix representations of superoperators
----------------------------------------

We can write the unconditional vacuum master equation
:math:`d\rho/dt=c\rho c^\dagger-\frac{1}{2}(c^\dagger c\rho+\rho c^\dagger c)`
as a system of coupled first-order ordinary differential equations:

.. math::

   \begin{align}
   \operatorname{Tr}[\Lambda^{jk}\Lambda^{jk}]\frac{\mathrm{d}\rho_{jk}}
   {\mathrm{d}t} &=\sum_{p,q}\rho_{pq}\left(\sum_{m,n}|c_{mn}|^2
   \operatorname{Tr}
   \left[\Lambda^{jk}\left(\Lambda^{mn}\Lambda^{pq}\Lambda^{mn}-
   \frac{1}{2}(\Lambda^{mn}\Lambda^{mn}\Lambda^{pq}+\Lambda^{pq}\Lambda^{mn}
   \Lambda^{mn})\right)\right]+\right. \\
   & \quad\left.\sum_{dm+n<dr+s}2\Re\left\{c_{mn}c_{rs}^*
   \operatorname{Tr}\left[\Lambda^{jk}\left(\Lambda^{mn}\Lambda^{pq}
   \Lambda^{rs}-\frac{1}{2}(\Lambda^{rs}\Lambda^{mn}\Lambda^{pq}+
   \Lambda^{pq}\Lambda^{rs}\Lambda^{mn})\right)\right]\right\}\right)
   \end{align}

This means I can write the vectorized version of the equation, using single
indices :math:`w=dr+s`, :math:`x=dj+k`, :math:`y=dp+q`, and :math:`z=dm+n` for
:math:`\vec{\rho}`:

.. math::

   \frac{d\vec{\rho}}{dt}=D(\vec{c})\vec{\rho}

The matrix :math:`D(\vec{c})` has entries:

.. math::

   \begin{align}
   D_{xy}(\vec{c}) &=(\operatorname{Tr}[\Lambda^x\Lambda^x])^{-1}\left(
   \sum_z|c_z|^2\operatorname{Tr}[\Lambda^x(\Lambda^z\Lambda^y\Lambda^z-
   \frac{1}{2}(\Lambda^z\Lambda^z\Lambda^y+
   \Lambda^y\Lambda^z\Lambda^z))]+\right. \\
   & \quad\left.\sum_{z>w}2\Re\left\{c_z c_w^*\operatorname{Tr}[\Lambda^x(
   \Lambda^z\Lambda^y\Lambda^w-\frac{1}{2}(\Lambda^w\Lambda^z\Lambda^y+
   \Lambda^y\Lambda^w\Lambda^z))]\right\}\right)
   \end{align}

In a similar way we can calculate:

.. math::

   \rho^\prime=\frac{M}{2}[c,[c,\rho]]+\frac{M^*}{2}[c^\dagger,[c^\dagger,\rho]]

in the vectorized form:

.. math::

   \vec{\rho}^\prime=E(M,\vec{c})\vec{\rho}

where :math:`E(M,\vec{c})` has entries:

.. math::

   \begin{align}
   E_{xy}(M,\vec{c})&=2(\operatorname{Tr}[\Lambda^x\Lambda^x])^{-1}
   \left(\sum_{w<z}\Re\{M^*c_wc_z\}\Re\{
   \operatorname{Tr}[\Lambda^x(\Lambda^w\Lambda^z\Lambda^y+
   \Lambda^y\Lambda^w\Lambda^z)-
   2\Lambda^x\Lambda^w\Lambda^y\Lambda^z]\}+\right. \\
   &\quad\left.\sum_w\Re\{M^*c_w^2\}(\Re\{
   \operatorname{Tr}[\Lambda^x\Lambda^w\Lambda^w\Lambda^y]\}-
   \operatorname{Tr}[\Lambda^x\Lambda^w\Lambda^y\Lambda^w])\right)
   \end{align}

If I vectorize the plant Hamiltonian:

.. math::

   \begin{align}
   H&=\sum_zh_z\Lambda^z,&h_z&\in\mathbb{R}
   \end{align}

I can then calculate:

.. math::

   \frac{d\rho}{dt}=-i[H,\rho]

in the vectorized form:

.. math::

   \frac{d\vec{\rho}}{dt}=F(\vec{h})\vec{\rho}

where :math:`F(\vec{h})` has entries:

.. math::

   F_{x,y}(\vec{h})=(\operatorname{Tr}[\Lambda^x\Lambda^x])^{-1}\sum_zh_z\,
   \Im\left\{\operatorname{Tr}[\Lambda^x(\Lambda^z\Lambda^y-
   \Lambda^y\Lambda^z)]\right\}

Nonlinear superoperator representation
--------------------------------------

The stochastic expression:

.. math::

   d\rho=dW\,(c\rho+\rho c^\dagger-\rho\operatorname{Tr}[(c+c^\dagger)\rho])

can be calculated:

.. math::

   d\vec{\rho}=dW(G+\vec{k}\cdot\vec{\rho})\vec{\rho}

where we define:

.. math::

   \begin{align}
   G_{x,y}&=2\left(\operatorname{Tr}[\Lambda^x\Lambda^x]\right)^{-1}\sum_z
   \Re\left\{c_z\operatorname{Tr}[\Lambda^x\Lambda^z\Lambda^y]\right\} \\
   k_x&=-2\Re\{c_x\}\operatorname{Tr}[\Lambda^y\Lambda^y]
   \end{align}
