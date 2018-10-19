from nose.tools import assert_almost_equal, assert_equal, assert_true
import pysme.gellmann as gm
import pysme.gramschmidt as gs
import pysme.sparse_system_builder as ssb
import pysme.system_builder as sb
import pysme.grid_conv as gc
import pysme.integrate as integrate
import pysme.matrix_form as mf
import pysme.hierarchy as hier

import numpy as np
import itertools as it
import sparse
from sparse import COO

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

def check_trace_preservation(supop_builder, supop_setup):
    """Make sure that the identity component of the density operator doesn't
    evolve.

    :param supop_builder:   Function that constructs a superoperator matrix
                            given the output from supop_setup.
    :param supop_setup:     Function that takes a coupling op and basis (minus
                            identity) as arguments and returns the arguments
                            needed for supop_builder to construct the
                            appropriate matrix.
    """

    for dim in range(2, 3 + 1):
        for row in range(dim):
            for col in range(dim):
                # Couple using all the different Gell-Mann matrices
                c_op = gm.gellmann(row + 1, col + 1, dim)
                kwargs = supop_setup(c_op, basis(dim)[:-1])
                D_matrix = supop_builder(**kwargs)
                for entry in D_matrix[-1]:
                    assert_almost_equal(0, entry, 7)

    for dim in range(2, 3 + 1):
        np.random.seed(dim)
        # Couple using random matrices
        rand_c_op = np.random.randn(dim, dim) + 1.j*np.random.randn(dim, dim)
        kwargs = supop_setup(rand_c_op, basis(dim)[:-1])
        D_matrix = sb.diffusion_op(**kwargs)
        for entry in D_matrix[-1]:
            assert_almost_equal(0, entry, 7)

def check_vectorize(operators, basis):
    vectorizations = [sb.vectorize(operator, basis) for operator in operators]
    reconstructions = [sum([coeff*basis_el for coeff, basis_el in
                            zip(vectorization, basis)]) for vectorization in
                       vectorizations]
    for operator, reconstruction in zip(operators, reconstructions):
        diff = reconstruction - operator
        assert_almost_equal(np.trace(np.dot(diff.conj().T, diff)), 0, 7)

def test_system_builder():
    check_trace_preservation(sb.diffusion_op, lambda c_op, partial_basis:
                             sb.op_calc_setup(c_op, 0, 0, np.zeros(c_op.shape),
                                              partial_basis))
    squeezing_params = [1, 1.j, -1, -1.j]
    for M in squeezing_params:
        check_trace_preservation(sb.double_comm_op, lambda c_op, partial_basis:
                                 sb.op_calc_setup(c_op, M, 0,
                                                  np.zeros(c_op.shape),
                                                  partial_basis))
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

    taylor_1_5_integrator = integrate.Taylor_1_5_HomodyneIntegrator(L, 0, 0,
            np.zeros(L.shape))
    t15_soln = taylor_1_5_integrator.integrate(rho_0, times)
    check_density_matrices(t15_soln)
    check_purities(t15_soln)

def test_against_matrix_implementation():
    r'''Compare an Euler trajectory computed naively using matrices to the
    Euler trajectory computed by our implementation for a particular
    realization of noise.

    '''
    c_loaded = np.load('tests/matrix_euler_test_0/coupling_op.npy')
    MN_arr = np.load('tests/matrix_euler_test_0/M_N.npy')
    M_loaded = MN_arr[0]
    N_loaded = MN_arr[1]
    H_loaded = np.load('tests/matrix_euler_test_0/H.npy')
    times_loaded = np.load('tests/matrix_euler_test_0/times.npy')
    U1s_loaded = np.load('tests/matrix_euler_test_0/U1s.npy')
    rho0_loaded = np.load('tests/matrix_euler_test_0/rho_0.npy')
    rhos_loaded = np.load('tests/matrix_euler_test_0/rhos.npy')

    test_integrator = integrate.EulerHomodyneIntegrator(c_loaded, M_loaded,
                                                        N_loaded, H_loaded)
    test_soln = test_integrator.integrate(rho0_loaded, times_loaded,
                                          U1s_loaded)
    test_rhos = test_soln.get_density_matrices()
    test_errors = test_rhos - rhos_loaded
    error_norms = [sb.norm_squared(test_errors[j])
                   for j in range(test_errors.shape[0])]
    assert_almost_equal(max(error_norms), 0.0, 7)

def test_against_t1_t2_matrix_euler_test_vector():
    with open('tests/t1-t2-matrix-euler-test-vector.npz', 'rb') as f:
        test_vec = np.load(f)
        Ls = test_vec['Ls']
        H = test_vec['H']
        rho0 = test_vec['rho0']
        times = test_vec['times']
        loaded_rhos = test_vec['rhos']
    integrator = integrate.UncondLindbladIntegrator(Ls, H)
    soln = integrator.integrate(rho0, times)
    rhos = soln.get_density_matrices()
    errors = rhos - loaded_rhos
    error_norms = [sb.norm_squared(errors[j])
                   for j in range(errors.shape[0])]
    # Normally I'm checking to 7 decimal places, but my Euler integrator
    # seems to only be able to get within 6 decimal places of the vectorized
    # solution. Maybe I'll bother to get an analytic expression one of these
    # days.
    assert_almost_equal(max(error_norms), 0.0, 6)

def test_against_random_spin1_matrix_euler_test_vector():
    with open('tests/random-spin1-matrix-euler-test-vector.npz', 'rb') as f:
        test_vec = np.load(f)
        Ls = test_vec['Ls']
        H = test_vec['H']
        rho0 = test_vec['rho0']
        times = test_vec['times']
        loaded_rhos = test_vec['rhos']
    integrator = integrate.UncondLindbladIntegrator(Ls, H)
    soln = integrator.integrate(rho0, times)
    rhos = soln.get_density_matrices()
    errors = rhos - loaded_rhos
    error_norms = [sb.norm_squared(errors[j])
                   for j in range(errors.shape[0])]
    # Normally I'm checking to 7 decimal places, but my Euler integrator
    # seems to only be able to get within 6 decimal places of the vectorized
    # solution. Analytic solution for a random instance like this is probably
    # not going to happen.
    assert_almost_equal(max(error_norms), 0.0, 6)

def check_sparse_vectorization(sparse_basis, ops):
    for op in ops:
        op_there_back = sparse_basis.matrize(sparse_basis.vectorize(op))
        assert_almost_equal(np.abs(op - op_there_back).max(), 0.0, 7)

def check_sparse_duals(sparse_basis, ops):
    for op1, op2 in it.product(ops, ops):
        tr = np.trace(op1.conj().T @ op2)
        ip = np.dot(sparse_basis.dualize(op1), sparse_basis.vectorize(op2))
        assert_almost_equal(np.abs(tr - ip), 0.0, 7)

def check_sparse_real_sand(sparse_basis, X, rho, Y):
    x_vec = sparse_basis.vectorize(X)
    y_vec = sparse_basis.vectorize(Y)
    rho_vec = sparse_basis.vectorize(rho)
    dense_real_sand = (X @ rho @ Y.conj().T + Y @ rho @ X.conj().T) / 2
    real_sand_matrix = sparse_basis.make_real_sand_matrix(x_vec, y_vec)
    real_sand_vec = COO.from_numpy(real_sand_matrix @ rho_vec.todense())
    assert_almost_equal(np.abs(dense_real_sand -
                               sparse_basis.matrize(real_sand_vec)).max(),
                        0.0, 7)

def check_sparse_real_comm(sparse_basis, X, rho, Y):
    x_vec = sparse_basis.vectorize(X)
    y_vec = sparse_basis.vectorize(Y)
    rho_vec = sparse_basis.vectorize(rho)
    dense_real_comm = (mf.comm(X @ rho, Y.conj().T) +
                       mf.comm(Y, rho @ X.conj().T)) / 2
    real_comm_matrix = sparse_basis.make_real_comm_matrix(x_vec, y_vec)
    real_comm_vec = COO.from_numpy(real_comm_matrix @ rho_vec.todense())
    assert_almost_equal(np.abs(dense_real_comm -
                               sparse_basis.matrize(real_comm_vec)).max(),
                        0.0, 7)

def check_sparse_hamil_comm(sparse_basis, H, rho):
    h_vec = sparse_basis.vectorize(H)
    rho_vec = sparse_basis.vectorize(rho)
    dense_hamil_comm = -1j * mf.comm(H, rho)
    hamil_comm_matrix = sparse_basis.make_hamil_comm_matrix(h_vec)
    hamil_comm_vec = COO.from_numpy(hamil_comm_matrix @ rho_vec.todense())
    assert_almost_equal(np.abs(dense_hamil_comm -
                               sparse_basis.matrize(hamil_comm_vec)).max(),
                        0.0, 7)

def check_trivial_construction():
    # Build some trivial integrators to make sure this edge case is handled
    # appropriately (in the past such a small factory resulted in dense arrays
    # that broke sparse.tensordot).
    Id_triv = np.eye(1, dtype=np.complex)
    zero_triv = np.zeros((1, 1), dtype=np.complex)
    S = Id_triv
    L = Id_triv # Important that this be Id and not 0
    H = Id_triv
    xi_fn = lambda t: 0
    factory = hier.HierarchyIntegratorFactory(1, 0)
    try:
        factory.make_euler_jump_integrator(xi_fn, S, L, H)
    except AttributeError as e:
        if e.args[0] == "'numpy.ndarray' object has no attribute 'tocsr'":
            assert_true(False, 'AttributeError "{}" suggests a dense array was '
                        'used when a sparse array was required.'.
                        format(e.args[0]))
        else:
            raise e

def test_sparse_system_builder():
    d = 30
    sparse_basis = ssb.SparseBasis(d)
    rand = np.random.RandomState(212114116)
    X = rand.standard_normal((d, d)) + 1.j * rand.standard_normal((d, d))
    Y = rand.standard_normal((d, d)) + 1.j * rand.standard_normal((d, d))
    rho_temp = rand.standard_normal((d, d)) + 1.j * rand.standard_normal((d, d))
    rho = rho_temp @ rho_temp.conj().T / np.trace(rho_temp @ rho_temp.conj().T)
    H_temp = rand.standard_normal((d, d)) + 1.j * rand.standard_normal((d, d))
    H = (H_temp + H_temp.conj().T) / 2
    check_sparse_vectorization(sparse_basis, [X, Y, rho, H])
    check_sparse_duals(sparse_basis, [X, Y, rho, H])
    check_sparse_real_sand(sparse_basis, X, rho, Y)
    check_sparse_real_comm(sparse_basis, X, rho, Y)
    check_sparse_hamil_comm(sparse_basis, H, rho)
    check_trivial_construction()
