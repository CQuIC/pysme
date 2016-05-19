from nose.tools import assert_almost_equal, assert_equal, assert_true
import pysme.gellmann as gm
import pysme.gramschmidt as gs
import pysme.system_builder as sb
import pysme.grid_conv as gc
import pysme.integrate as integrate
import numpy as np

def check_orthogonal(A, B):
    dot_prod = np.sqrt(np.trace(np.dot(A.conj().T, B)))
    assert_almost_equal(dot_prod, 0.0, 7)

def check_hermitian(A):
    non_herm = A - A.conj().T
    non_herm_norm = np.sqrt(np.trace(np.dot(non_herm.conj().T, non_herm)))
    assert_almost_equal(non_herm_norm, 0.0, 7)

def check_traceless(A):
    assert_almost_equal(np.trace(A), 0.0, 7)

def check_norm(A, norm):
    assert_almost_equal(np.sqrt(np.trace(np.dot(A.conj().T, A))), norm, 7)

def test_gellmann():
    for d in range(1, 5 + 1):
        matrices = [ [ gm.gellmann(j, k, d) for k in range(1, d + 1) ] for j in
            range(1, d + 1) ]
        for j in range(1, d + 1):
            for k in range(1, d + 1):
                check_hermitian(matrices[j - 1][k - 1])
                if j != d or k != d:
                    check_traceless(matrices[j - 1][k - 1])
                    check_norm(matrices[j - 1][k - 1], np.sqrt(2))
                else:
                    check_norm(matrices[j - 1][k - 1], np.sqrt(d))
                for jj in range(1, d + 1):
                    for kk in range(1, d + 1):
                        if jj != j or kk != k:
                            check_orthogonal(matrices[j - 1][k - 1],
                                matrices[jj - 1][kk - 1])

def check_recon(A, basis):
    A_coeffs = [ np.trace(np.dot(vect, A)) for vect in basis[0:3] ]
    A_recon = sum([ coeff*vect for coeff, vect in zip(A_coeffs, basis) ])
    diff = A - A_recon
    assert_almost_equal(np.sqrt(np.trace(np.dot(diff.conj().T, diff))), 0, 7)

def check_mat_eq(A, B):
    diff = A - B
    assert_almost_equal(np.sqrt(np.trace(np.dot(diff.conj().T, diff))), 0, 7)

def test_gramschmidt():
    test_matrices = [
        np.array([[ 5. + 0.j,  1. + 0.j],
                  [-2. + 0.j,  5. + 1.j]]),
        np.array([[ 1. + 0.j,  0. + 1.j],
                  [ 0. + 1.j, -1. + 0.j]]),
        np.array([[ 1. + 0.j,  1. + 1.j],
                  [-1. - 1.j,  1. + 0.j]]),
        np.array([[ 0. + 1.j,  1. + 0.j],
                  [ 1. + 0.j,  0. + 0.j]]),
        np.array([[ 1. + 0.j,  1. + 0.j,  0. + 0.j,  0. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  1. + 0.j,  0. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  0. + 0.j,  1. + 0.j],
                  [ 0. + 0.j,  0. + 0.j,  0. + 0.j,  0. + 0.j]]),
        ]

    for test_matrix in test_matrices:
        d = test_matrix.shape[0]
        basis = gs.orthonormalize(test_matrix)
        check_recon(test_matrix, basis)
        for m in range(len(basis)):
            if m == 0:
                check_mat_eq(basis[m], np.eye(d)/np.sqrt(d))
            else:
                check_traceless(basis[m])
            check_hermitian(basis[m])
            for n in range(len(basis)):
                if m != n:
                    check_orthogonal(basis[m], basis[n])
                else:
                    check_norm(basis[m], 1)

def basis(n):
    """Return an orthogonal basis for n-by-n operators, with the identity in the
    last position.

    """

    return [gm.gellmann(j, k, n) for j in range(1, n + 1) for k in
            range(1, n + 1)]

def check_trace_preservation(supop_builder):
    """Make sure that the identity component of the density operator doesn't
    evolve.

    :param supop_builder:   Function that takes a coupling op and basis (minus
                            identity) as arguments and returns the appropriate
                            matrix
    """

    for dim in range(2, 3 + 1):
        for row in range(dim):
            for col in range(dim):
                # Couple using all the different Gell-Mann matrices
                c_op = gm.gellmann(row + 1, col + 1, dim)
                D_matrix = supop_builder(c_op, basis(dim)[:-1])
                for entry in D_matrix[-1]:
                    assert_equal(0, entry)

    for dim in range(2, 3 + 1):
        np.random.seed(dim)
        # Couple using random matrices
        rand_c_op = np.random.randn(dim, dim) + 1.j*np.random.randn(dim, dim)
        D_matrix = sb.diffusion_op(rand_c_op, basis(dim)[:-1])
        for entry in D_matrix[-1]:
            assert_equal(0, entry)

def check_vectorize(operators, basis):
    vectorizations = [sb.vectorize(operator, basis) for operator in operators]
    reconstructions = [sum([coeff*basis_el for coeff, basis_el in
                            zip(vectorization, basis)]) for vectorization in
                       vectorizations]
    for operator, reconstruction in zip(operators, reconstructions):
        diff = reconstruction - operator
        assert_almost_equal(np.trace(np.dot(diff.conj().T, diff)), 0, 7)

def test_system_builder():
    check_trace_preservation(sb.diffusion_op)
    squeezing_params = [1, 1.j, -1, -1.j]
    for M in squeezing_params:
        check_trace_preservation(lambda c, b: sb.double_comm_op(c, M, b))
    c2_operators = [np.array([[0, 1],
                              [0, 0]]),
                    np.array([[0, 0],
                              [1, 0]]),
                    np.array([[1, 0],
                              [0, -1]]),
                    np.array([[0, 1],
                              [1, 0]]),
                    np.array([[0, -1.j],
                              [1.j, 0]]),
                    np.array([[1, 2],
                              [3, 4]])]

    c3_operators = [np.array([[0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 0]]),
                    np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0]]),
                    np.array([[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, -1]]),
                    np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])]
    check_vectorize(c2_operators, basis(2))
    # Check vectorization of a mixing of the basis vectors (designed to catch a
    # previous error in calculating normalization)
    orth_mat = 0.5*np.array([[1, 1, 1, 1],
                             [1, 1, -1, -1],
                             [1, -1, 1, -1],
                             [1, -1, -1, 1]])
    mixed_basis2 = [sum([entry*basis_el for entry, basis_el in
                         zip(row, basis(2))]) for row in orth_mat]
    check_vectorize(c2_operators, mixed_basis2)
    check_vectorize(c3_operators, basis(3))

def new_dW_from_dW(dW_2n, dW_2n_1):
    return dW_2n + dW_2n_1

def new_dW_from_U(new_U1_n, new_dt):
    return new_U1_n*np.sqrt(new_dt)

def new_dZ_from_dZ(dZ_2n, dZ_2n_1, dW_2n, dt):
    return dZ_2n + dt*dW_2n + dZ_2n_1

def new_dZ_from_U(new_U1_n, new_U2_n, new_dt):
    return new_dt**(3/2)*(new_U1_n + new_U2_n/np.sqrt(3))/2

def test_double_increments():
    r'''Make sure increments are doubled so that

    .. math::

       \tilde{\Delta}&=2\Delta \\
       \Delta\tilde{W}_n&=\Delta W_{2n}+\Delta W_{2n+1} \\
       &=\tilde{U}_{1,n}\sqrt{\tilde{\Delta}} \\
       \Delta\tilde{Z}_n&=\Delta Z_{2n}+\Delta_{2n+1}\Delta W_{2n}+
       \Delta Z_{2n+1}

    '''
    np.random.seed(1472)
    dt = 0.1
    steps = 10
    times = np.linspace(0, steps*dt, steps + 1)
    U1s = np.random.randn(steps)
    U2s = np.random.randn(steps)
    dWs = U1s*np.sqrt(dt)
    dZs = dt**(3/2)*(U1s + U2s/np.sqrt(3))/2
    new_times, new_U1s, new_U2s = gc.double_increments(times, U1s, U2s)
    new_dt = new_times[1] - new_times[0]
    assert_almost_equal(2*dt, new_dt, 7)
    for n in range(steps//2):
        assert_almost_equal(new_dW_from_dW(dWs[2*n], dWs[2*n+1]),
                            new_dW_from_U(new_U1s[n], new_dt), 7)
        assert_almost_equal(new_dZ_from_dZ(dZs[2*n], dZs[2*n+1], dWs[2*n], dt),
                            new_dZ_from_U(new_U1s[n], new_U2s[n], new_dt), 7)

def check_convergence_rate(expected_rate, integrator, rho_0, times, U1s_arr,
                           U2s_arr):
    rates = [gc.calc_rate(integrator, rho_0, times, U1s, U2s)
             for U1s, U2s in zip(U1s_arr, U2s_arr)]
    average_rate = np.mean(np.array(rates))
    assert_true(expected_rate - 0.25 < average_rate < expected_rate + 0.25,
                'Average convergence rate {0} is outside of ({1}, {2})'.format(
                    average_rate, expected_rate - 0.25, expected_rate + 0.25))

def test_convergence():
    r'''Estimate convergence rates for different integrators to see if they
    match expected behavior.

    '''

    trajectories = 128
    times = np.linspace(0, 1, 65)
    increments = len(times) - 1
    np.random.seed(78543287)
    U1s_arr = np.random.randn(trajectories, increments)
    U2s_arr = np.random.randn(trajectories, increments)

    # Pauli matrices
    X = np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]])
    Y = np.array([[0. + 0.j, 0. - 1.j], [0. + 1.j, 0. + 0.j]])
    Z = np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, -1 + 0.j]])
    Id = np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, 1. + 0.j]])
     
    # Evolution parameters
    H = 1*X
    g = 1
    L = np.sqrt(g)*(X - 1.j*Y)/2
     
    # Initial Bloch vector parameters
    r = 1
    theta = 0
    phi = 0
    rho_0 = 0.5*(Id + r*(np.cos(theta)*Z +
                 np.sin(theta)*(np.cos(phi)*X + np.sin(phi)*Y)))

    milstein_integrator = integrate.MilsteinHomodyneIntegrator(L, 0, 0, H)
    taylor_1_5_integrator = integrate.Taylor_1_5_HomodyneIntegrator(L, 0, 0, H)

    check_convergence_rate(1, milstein_integrator, rho_0, times, U1s_arr,
                           U2s_arr)
    check_convergence_rate(1.5, taylor_1_5_integrator, rho_0, times, U1s_arr,
                           U2s_arr)

def check_density_matrices(solution):
    density_matrices = solution.get_density_matrices()
    non_herm = [rho - rho.conj().T for rho in density_matrices]
    assert_almost_equal(max([np.trace(np.dot(A.conj().T, A)).real
                             for A in non_herm]), 0, 7)

def check_purities(solution):
    density_matrices = solution.get_density_matrices()
    calc_purities = [np.trace(np.dot(rho.conj().T, rho)).real
                     for rho in density_matrices]
    returned_purities = solution.get_purities()
    assert_almost_equal(np.max(np.abs(calc_purities - returned_purities)), 0, 7)

def test_solution_functions():
    r'''Check to see if convenience functions provided by the Solution object
    are behaving correctly.

    '''
    X = np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]])
    Y = np.array([[0. + 0.j, 0. - 1.j], [0. + 1.j, 0. + 0.j]])
    Z = np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, -1 + 0.j]])
    Id = np.array([[1. + 0.j, 0. + 0.j], [0. + 0.j, 1. + 0.j]])
    L = (X - 1.j*Y)/2
    rho_0 = (Id + Z)/2
    times = np.linspace(0, 1, 65)

    milstein_integrator = integrate.MilsteinHomodyneIntegrator(L, 0, 0,
            np.zeros(L.shape))
    solution = milstein_integrator.integrate(rho_0, times)
    check_density_matrices(solution)
    check_purities(solution)

def test_against_matrix_implementation():
    r'''Compare an Euler trajectory computed naively using matrices to the
    Euler trajectory computed by our implementation for a particular
    realization of noise.

    '''
    c_loaded = np.load('matrix_euler_test_0/coupling_op.npy')
    MN_arr = np.load('matrix_euler_test_0/M_N.npy')
    M_loaded = MN_arr[0]
    N_loaded = MN_arr[1]
    H_loaded = np.load('matrix_euler_test_0/H.npy')
    times_loaded = np.load('matrix_euler_test_0/times.npy')
    U1s_loaded = np.load('matrix_euler_test_0/U1s.npy')
    rho0_loaded = np.load('matrix_euler_test_0/rho_0.npy')
    rhos_loaded = np.load('matrix_euler_test_0/rhos.npy')

    test_integrator = integrate.EulerHomodyneIntegrator(c_loaded, M_loaded,
                                                        N_loaded, H_loaded)
    test_soln = test_integrator.integrate(rho0_loaded, times_loaded,
                                          U1s_loaded)
    test_rhos = test_soln.get_density_matrices()
    test_errors = test_rhos - rhos_loaded
    error_norms = [sb.norm_squared(test_errors[j])
                   for j in range(test_errors.shape[0])]
    assert_almost_equal(max(error_norms), 0.0, 7)
