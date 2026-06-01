# =============================================================================
# FIȘIERUL 1: svd_core.py
# =============================================================================
"""
Modul Core pentru Analiză Numerică
Implementare SVD (Descompunerea la Valori Singulare) fără funcții predefinite.
Bazează pe reflexii Householder și Iterația QR.
"""

import numpy as np
import time

def QR_Householder(A):
    """
    Factorizare QR folosind reflexii Householder.
    """
    m, n = A.shape
    R = np.copy(A)
    Q = np.eye(m)
    for j in range(n):
        x = R[j:, j]
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-15:
            v = np.copy(x)
            v[0] += np.sign(x[0]) * norm_x if x[0] != 0 else norm_x
            v = v / np.linalg.norm(v)
            R[j:, j:] -= 2.0 * np.outer(v, np.dot(v, R[j:, j:]))
            Q[:, j:] -= 2.0 * np.outer(np.dot(Q[:, j:], v), v)
    return Q[:, :n], R[:n, :]

def Tridiag_Householder(A):
    """
    Aducerea unei matrici simetrice la forma tridiagonală folosind Householder.
    """
    n = np.shape(A)[0] 
    T_mat = np.copy(A)
    Q = np.eye(n)
    for k in range(n - 2):
        v = np.copy(T_mat[k + 1:, k]) 
        norm_v = np.linalg.norm(v) 
        if norm_v > 1e-15:
            u = np.copy(v)
            u[0] += np.sign(v[0]) * norm_v if v[0] != 0 else norm_v
            u = u / np.linalg.norm(u)
            H_mica = np.eye(n - k - 1) - 2.0 * np.outer(u, u)
            H = np.eye(n)
            H[k + 1:, k + 1:] = H_mica
            T_mat = H @ T_mat @ H 
            Q = Q @ H 
    return Q, T_mat

def QR_iteration(A, Q, TOL=1e-4):
    """
    Iterația QR pentru diagonalizarea unei matrici tridiagonale.
    """
    T_mat = Q.T @ A @ Q
    V = Q
    if T_mat.shape[0] <= 1: 
        return T_mat, V
    max_iter = 500
    it = 0
    while np.max(np.abs(np.diag(T_mat, k=-1))) > TOL and it < max_iter:
        Q1, R1 = QR_Householder(T_mat)
        T_mat = R1 @ Q1 
        V = V @ Q1
        it += 1
    return T_mat, V

def SVD_Custom(A):
    """
    SVD de la zero.
    Metodă: Diagonalizarea matricei de covarianță M = A^T * A.
    Returnează: U (vectori singulari stângi), S (valori singulare), Vt (vectori drepți transpuși).
    """

    A = A.astype(np.float64)

    m, n = A.shape
    M = A.T @ A  # Matrice simetrică pozitiv semidefinită
    
    Q0, T0 = Tridiag_Householder(M)
    D, V = QR_iteration(M, Q0)

    eigenvalues = np.diag(D).copy()
    eigenvalues[eigenvalues < 0] = 0  # Evităm nan-uri din erori numerice FP
    
    # Sortare descrescătoare
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]
    
    S = np.sqrt(eigenvalues)
    
    # Calcul U pe baza V și S
    U = np.zeros((m, n))
    for i in range(n):
        if S[i] > 1e-10: 
            U[:, i] = (A @ V[:, i]) / S[i]
        else: 
            U[:, i] = 0
            
    return U, S, V.T

def benchmark_svd(A):
    """
    Funcție de validare academică a implementării SVD.
    Compară timpul de execuție și precizia față de numpy.linalg.svd.
    """
    print("\n" + "="*50)
    print(" BAREM - BENCHMARKING SVD CUSTOM VS NUMPY")
    print("="*50)
    
    # 1. Timp Custom SVD
    start_time = time.time()
    U_c, S_c, Vt_c = SVD_Custom(A)
    time_custom = time.time() - start_time
    
    # Reconstrucție Custom
    A_rec_c = U_c @ np.diag(S_c) @ Vt_c
    err_custom = np.linalg.norm(A - A_rec_c, ord='fro')
    
    # 2. Timp Numpy SVD
    start_time = time.time()
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    time_np = time.time() - start_time
    
    # Reconstrucție Numpy
    A_rec_np = U_np @ np.diag(S_np) @ Vt_np
    err_np = np.linalg.norm(A - A_rec_np, ord='fro')
    
    # Formatare Tabel
    print(f"{'Metodă':<15} | {'Timp (s)':<10} | {'Eroare Reconstrucție (Frob)'}")
    print("-" * 50)
    print(f"{'Custom (QR)':<15} | {time_custom:<10.4f} | {err_custom:.4e}")
    print(f"{'Numpy (LAPACK)':<15} | {time_np:<10.4f} | {err_np:.4e}")
    print("="*50 + "\n")
    
    return U_c, S_c, Vt_c
