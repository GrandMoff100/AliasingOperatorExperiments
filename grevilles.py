import numpy as np

rank = 300
M = 5000
N = 1000
a = np.random.rand(M, rank)
b = np.random.rand(N, rank)

A = np.sum([np.outer(a[:, k], b[:, k]) for k in range(rank)], axis=0)
assert A.shape == (M, N)
assert np.linalg.matrix_rank(A) == rank
print("Calculating pseudoinverse using NumPy...")
A_plus_actual = np.linalg.pinv(A)
assert np.allclose(A @ A_plus_actual @ A, A)
assert np.allclose(A_plus_actual @ A @ A_plus_actual, A_plus_actual)


# Greville's algorithm for pseudoinverse
def greville_pseudoinverse(A, zero_tol: float = 1e-10):
    m, n = A.shape
    A_plus = np.zeros((n, m))
    if np.linalg.norm(A[:, 0]) > zero_tol:
        A_plus[0, :] = A[:, 0].T / np.sum(A[:, 0] ** 2)
    for j in range(1, n):
        d_j = A_plus[:j, :] @ A[:, j]
        c_j = A[:, j] - A[:, :j] @ d_j
        if np.linalg.norm(c_j) > zero_tol:  # column is independent
            b_j = c_j / np.sum(c_j ** 2)
        else: # Column in linearly dependent
            b_j = d_j @ A_plus[:j, :] / (1 + np.sum(d_j ** 2))
        A_plus[:j, :] -= np.outer(d_j, b_j)
        A_plus[j, :] = b_j
    return A_plus

print("Calculating pseudoinverse using Grevilles Algorithm...")
grevilles_A_plus = greville_pseudoinverse(A)
assert np.allclose(A @ grevilles_A_plus @ A, A)
assert np.allclose(grevilles_A_plus @ A @ grevilles_A_plus, grevilles_A_plus)
assert np.allclose(grevilles_A_plus, A_plus_actual)