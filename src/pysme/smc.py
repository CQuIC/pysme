"""
.. py:module:: smc.py
   :synopsis: Do sequential Monte Carlo inference using qinfer.
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

"""

import numpy as np
import pysme.integrate as smeint
import pysme.system_builder as sb
import pysme.gellmann as gm

# Don't want qinfer to be a required dependency
try:
    import qinfer as qi
except ImportError:
    import warnings
    warnings.warn(
        "Could not import qinfer. "
        "Sequential Monte Carlo support will be disabled."
        )
    qi = None

def precomp_fn(coupling_op, M_sq, N, H0, partial_basis, **kwargs):
    common_dict = sb.op_calc_setup(coupling_op, M_sq, N, H0, partial_basis)
    D_c = sb.diffusion_op(**common_dict)
    conjugate_dict = common_dict.copy()
    conjugate_dict['C_vector'] = common_dict['C_vector'].conjugate()
    D_c_dag = sb.diffusion_op(**conjugate_dict)
    E = sb.double_comm_op(**common_dict)
    F0 = sb.hamiltonian_op(**common_dict)

    Q_minus_F = (N + 1) * D_c + N * D_c_dag + E
    G, k_T = sb.wiener_op(**common_dict)

    return_vals = {'Q_minus_F': Q_minus_F,
                   'F0': F0,
                   'diffusion_reps': {'G': G, 'k_T': k_T}}
    more_vals = {'c_op': coupling_op, 'M_sq': M_sq, 'N': N,
                 'H': H0, 'partial_basis': partial_basis}
    return_vals.update(more_vals)
    return return_vals

def parameter_fn(B, precomp_data):
    drift_rep = precomp_data['Q_minus_F'] + B * precomp_data['F0']
    constructor_kwargs = precomp_data.copy()
    constructor_kwargs.update({'drift_rep': drift_rep})
    return constructor_kwargs

class HomodyneQubitPrecessionModel(qi.Model):
    '''This is a `qinfer` `Model` for replicating the work of Chase and Geremia
    in *Single shot parameter estimation via continuous quantum measurement*,
    arXiv:`0811.0601 <https://arxiv.org/abs/0811.0601>`_.

    '''

    def __init__(self, L, H0):
        # The `IntegratorFactory` returns an integrator appropriate for the
        # given modelparams.
        super(HomodyneQubitPrecessionModel, self).__init__()
        basis = gm.get_basis(2)
        precomp_args = {'coupling_op': L,
                        'M_sq': 0,
                        'N': 0,
                        'H0': H0,
                        'partial_basis': basis[:-1]}
        precomp_data = precomp_fn(**precomp_args)
        self.integrator_factory = \
            smeint.IntegratorFactory(smeint.TrDecMilsteinHomodyneIntegrator,
                                     precomp_data, parameter_fn)
        self.traceless_basis = \
                self.integrator_factory.precomp_data['partial_basis']
        self.drifted_particles = {}

    @property
    def n_modelparams(self):
        # The first number is the magnetic field strength, and the remaining 3
        # numbers are the real coefficients for the traceless basis operator
        # representation of rho
        return 4

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        # Patently false for continuous outcome space, but don't think it's used
        # for what I currently care about.
        return 2

    def are_models_valid(self, modelparams):
        # Let's only think about positive magnetic field strengths.  For our
        # qubit parametrization, all Bloch vector components must be between
        # -0.5 and 0.5.
        return np.logical_and(modelparams[:,0] >= 0,
                              np.all(np.logical_and(modelparams[:,1:] >= -0.5,
                                                    modelparams[:,1:] <= 0.5),
                                     axis=1))

    @property
    def expparams_dtype(self):
        # The experiment is defined by a sequence of times that the integrated
        # homodyne current is known at, which requires us to use `object` since
        # these can be variable length.
        return [('times', 'object')]

    def likelihood(self, outcomes, modelparams, expparams):
        Id = np.eye(2)
        L = np.empty((outcomes.shape[0], modelparams.shape[0],
                      expparams.shape[0]))
        self.drifted_particles = {}
        for i, dMs in enumerate(outcomes):
            for j, modelparam in enumerate(modelparams):
                B = modelparam[0]
                rho_0_vec = modelparam[1:]
                for k, expparam in enumerate(expparams):
                    times = expparam['times']
                    rho_0 = sum([comp * basis_vec
                                 for comp, basis_vec
                                 in zip(rho_0_vec, self.traceless_basis)] +
                                [Id / 2])
                    integrator = self.integrator_factory.make_integrator(B)
                    soln = integrator.integrate_measurements(rho_0, times, dMs)
                    # The trace of the density matrix for this trace-decreasing
                    # evolution is proportional to the likelihood.
                    likelihood = soln.get_expectations(Id)[-1]
                    L[i,j,k] = likelihood
                    # Renormalize the final state and store it in this object to
                    # be used by `update_timestep`.
                    rho_f = soln.vec_soln[-1,:-1] / likelihood
                    self.drifted_particles[B] = rho_f
        return L

    def update_timestep(self, modelparams, expparams):
        # Assume only one set of expparams provided.
        updated_modelparams = np.empty((modelparams.shape[0],
                                        modelparams.shape[1],
                                        expparams.shape[0]))
        for i, modelparam in enumerate(modelparams):
            # The magnetic field strength doesn't evolve with
            # the measurement record
            B = modelparam[0]
            updated_modelparams[i,0,:] = B
            updated_modelparams[i,1:,:] = self.drifted_particles[B][:,
                                                                    np.newaxis]

        return updated_modelparams

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        # Code complains when I don't have this.
        pass

    def domain(self, expparams):
        # Code complains when I don't have this.
        pass
