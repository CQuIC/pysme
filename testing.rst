.. Discussion how to test the stochastic integrators.

Testing
=======

To test the stochastic integrators, I am taking my cue from `Ian Hawke`_ and
employing *grid convergence*. There are two points that make using grid
convergence checks on stochastic integration slightly less trivial than for
ordinary integration.

.. _Ian Hawke: http://nbviewer.ipython.org/github/IanHawke/close-enough-balloons/blob/master/00-Close-Enough-Post-Overall.ipynb

The first point is that the stochastic integration methods have convergence
rates that are given as *expectation values* of the convergence rates for each
trajectory. For strong approximation techniques (which are supposed to
converge in trajectory), this means that I'll need to calculate the
convergence rates for an ensemble of trajectories and take the average in order
to compare to the expected convergence rate.

The second point is that I have to use consistent random increments
:math:`\Delta W` and :math:`\Delta Z` for each trajectory. I will do this by
calculating all my increments for the smallest timestep, and then using those
values to construct the corresponding increments for larger timesteps. My
integrators also deal in standard-normal random variable :math:`U_1` and
:math:`U_2`, so the thing I actually need to construct are the corresponding
standard-normal random variables :math:`\tilde{U}_1` and :math:`\tilde{U}_2`
for the larger time increments.

Longer stochastic increments
----------------------------

Let's write down what we want first:

.. math::

   \begin{align}
   \tilde{\Delta}&=2\Delta \\
   \Delta W&=U_1\sqrt{\Delta} \\
   \Delta\tilde{W}&=\tilde{U}_1\sqrt{\tilde{\Delta}} \\
   \Delta Z&=\frac{1}{2}\Delta^{3/2}\left(U_1+\frac{1}{\sqrt{3}}U_2\right) \\
   \Delta\tilde{Z}&=\frac{1}{2}\tilde{\Delta}^{3/2}\left(\tilde{U}_1
                   +\frac{1}{\sqrt{3}}\tilde{U}_2\right)\,.
   \end{align}

Now we will write down how the increments are defined and work out what our
new standard-normal random variables are.

.. math::

   \begin{align}
   \Delta_n&:=\int_{\tau_n}^{\tau_{n+1}}ds \\
   \tilde{\Delta}_n&:=\int_{\tau_n}^{\tau_{n+2}}ds \\
   &=\int_{\tau_n}^{\tau_{n+1}}ds+\int_{\tau_{n+1}}^{\tau_{n+2}}ds \\
   &=\Delta_n+\Delta_{n+1} \\
   \Delta W_n&:=\int_{\tau_n}^{\tau_{n+1}}dW_s \\
   \Delta \tilde{W}_n&:=\int_{\tau_n}^{\tau_{n+2}}dW_s \\
   &=\int_{\tau_n}^{\tau_{n+1}}dW_s+\int_{\tau_{n+1}}^{\tau_{n+2}}dW_s \\
   &=\Delta W_n+\Delta W_{n+1} \\
   \Delta Z_n&:=\int_{\tau_n}^{\tau_{n+1}}\int_{\tau_n}^{s_2}dW_{s_1}ds_2 \\
   \Delta\tilde{Z}_n&:=\int_{\tau_n}^{\tau_{n+2}}
   \int_{\tau_n}^{s_2}dW_{s_1}ds_2 \\
   &=\int_{\tau_n}^{\tau_{n+1}}\int_{\tau_n}^{s_2}dW_{s_1}ds_2
   +\int_{\tau_{n+1}}^{\tau_{n+2}}\int_{\tau_n}^{s_2}dW_{s_1}ds_2 \\
   &=\Delta Z_n
   +\int_{\tau_{n+1}}^{\tau_{n+2}}\int_{\tau_n}^{\tau_{n+1}}dW_{s_1}ds_2
   +\int_{\tau_{n+1}}^{\tau_{n+2}}\int_{\tau_{n+1}}^{s_2}dW_{s_1}ds_2 \\
   &=\Delta Z_n+\Delta_{n+1}\Delta W_n+\Delta Z_{n+1}\,.
   \end{align}

We will assume equal time intervals, so :math:`\Delta_n=\Delta`. We start by
assuming we are simulating :math:`\Delta W_n` and :math:`\Delta Z_n` by the
independent standard-normal random variables :math:`U_{1,n}` and :math:`U_{2,n}`
using the expressions

.. math::

   \begin{align}
   \Delta W_n&=U_{1,n}\sqrt{\Delta} \\
   \Delta Z_n&=\frac{1}{2}\Delta^{3/2}\left(U_{1,n}+\frac{1}{\sqrt{3}}U_{2,n}
   \right)\,.
   \end{align}

Now we want to derive expressions for the new independent standard-normal random
variables :math:`\tilde{U}_{1,n}` and :math:`\tilde{U}_{2,n}`. Start by
looking at :math:`\Delta\tilde{W}_n`:

.. math::

   \begin{align}
   \Delta\tilde{W}_n&=\Delta W_n+\Delta W_{n+1} \\
   &=(U_{1,n}+U_{1,n+1})\sqrt{\Delta} \\
   &=\frac{U_{1,n}+U_{1,n+1}}{\sqrt{2}}\sqrt{\tilde{\Delta}}\,.
   \end{align}

This tells us

.. math::

   \begin{align}
   \tilde{U}_{1,n}&=\frac{U_{1,n}+U_{1,n+1}}{\sqrt{2}}\,.
   \end{align}

It is easy to verify this is a standard-normal random variable.

Now look at :math:`\Delta\tilde{Z}_n`:

.. math::

   \begin{align}
   \Delta\tilde{Z}_n&=\Delta Z_n+\Delta\Delta W_n+\Delta Z_{n+1} \\
   &=\frac{1}{2}\Delta^{3/2}\left(U_{1,n}+\frac{1}{\sqrt{3}}U_{2,n}\right)
   +\Delta U_{1,n}\sqrt{\Delta}
   +\frac{1}{2}\Delta^{3/2}\left(U_{1,n+1}+\frac{1}{\sqrt{3}}U_{2,n+1}\right) \\
   &=\frac{1}{2}\Delta^{3/2}\left(3U_{1,n}+U_{1,n+1}+\frac{1}{\sqrt{3}}\left(
   U_{2,n}+U_{2,n+1}\right)\right) \\
   &=\frac{1}{2}\tilde{\Delta}^{3/2}\frac{1}{2\sqrt{2}}\left(
   2(U_{1,n}+U_{1,n+1})+U_{1,n}-U_{1,n+1}
   +\frac{1}{\sqrt{3}}(U_{2,n}+U_{2,n+1})\right) \\
   &=\frac{1}{2}\tilde{\Delta}^{3/2}\left(\frac{U_{1,n}+U_{1,n+1}}{\sqrt{2}}
   +\frac{1}{\sqrt{3}}\left(\frac{\sqrt{3}}{2}\frac{U_{1,n}-U_{1,n+1}}{\sqrt{2}}
   +\frac{1}{2}\frac{U_{2,n}+U_{2,n+1}}{\sqrt{2}}\right)\right)\,.
   \end{align}

This tells us

.. math::

   \begin{align}
   \tilde{U}_{2,n}&=\frac{\sqrt{3}}{2}\frac{U_{1,n}-U_{1,n+1}}{\sqrt{2}}
   +\frac{1}{2}\frac{U_{2,n}+U_{2,n+1}}{\sqrt{2}}\,.
   \end{align}

Again, it is relatively straightforward to verify that this is another
standard-normal random variable and independent of :math:`\tilde{U}_{1,n}`.
