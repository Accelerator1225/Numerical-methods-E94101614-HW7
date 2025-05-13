
import numpy as np

# 系統矩陣與向量
A = np.array([
    [1, -1,  0, -1,  0,  0],
    [-1, 1, -1,  0, -1,  0],
    [0, -1, 1,  0,  1, -1],
    [-1, 0,  0,  1, -1, -1],
    [0, -1, 0,  1,  1, -1],
    [0,  0, -1, 0, -1,  1]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# Jacobi Method
def jacobi(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# Gauss-Seidel Method
def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i,:i], x_new[:i]) - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# SOR Method
def sor(A, b, x0, omega=1.2, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i,:i], x_new[:i]) + np.dot(A[i,i+1:], x[i+1:])
            x_new[i] = x[i] + omega * (b[i] - sigma - A[i,i]*x[i]) / A[i,i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# Conjugate Gradient Method
def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            return x_new, k+1
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        x, r = x_new, r_new
    return x, max_iter

# 初始猜測
x0 = np.zeros_like(b)

# 執行方法
jacobi_sol, jacobi_iter = jacobi(A, b, x0)
gs_sol, gs_iter = gauss_seidel(A, b, x0)
sor_sol, sor_iter = sor(A, b, x0, omega=1.2)
cg_sol, cg_iter = conjugate_gradient(A, b)

# 輸出結果
print("Q1: Jacobi Method")
print("Final Iteration:", jacobi_iter)
print("Result (may diverge):", jacobi_sol)
print()

print("Q2: Gauss-Seidel Method")
print("Final Iteration:", gs_iter)
print("Result (may diverge):", gs_sol)
print()

print("Q3: SOR Method (ω = 1.2)")
print("Final Iteration:", sor_iter)
print("Result (may diverge):", sor_sol)
print()

print("Q4: Conjugate Gradient Method")
print("Final Iteration:", cg_iter)
print("Result:", cg_sol)
print()
