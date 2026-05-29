# =============================================================================
# FIȘIERUL 3: dpa_pipeline.py
# =============================================================================
"""
Pipeline-ul complet pentru Atacul DPA cu Truncated SVD.
Rulează benchmark-ul, filtrează datele, atacă algoritmul AES și plotează graficele.
"""

import numpy as np
import matplotlib.pyplot as plt
from svd_core import benchmark_svd, SVD_Custom
from ascad_handler import extract_poi, AES_SBOX, hw_vec

def calculate_cpa(traces, plaintexts):
    """
    Atacul DPA folosind Corelația Pearson (CPA).
    Atacă primul octet din cheie (index 0).
    """
    num_traces, num_samples = traces.shape
    correlations = np.zeros((256, num_samples))
    
    # Normalizăm urmele pentru calcul mai rapid al corelației
    traces_mean = np.mean(traces, axis=0)
    traces_centered = traces - traces_mean
    traces_std = np.std(traces_centered, axis=0)
    traces_std[traces_std == 0] = 1 # Prevenim împărțirea la zero
    
    print("[CPA] Rulez corelația pentru toate cele 256 de chei posibile...")
    for key_guess in range(256):
        # 1. Generăm modelul ipotetic (Hamming Weight al ieșirii SBox)
        sbox_in = plaintexts[:, 0] ^ key_guess
        sbox_out = AES_SBOX[sbox_in]
        hyp_hw = hw_vec(sbox_out).astype(float)
        
        # 2. Pearson Correlation între HW ipotetic și fiecare punct în timp
        hyp_centered = hyp_hw - np.mean(hyp_hw)
        hyp_std = np.std(hyp_centered)
        
        if hyp_std > 0:
            cov = np.sum(traces_centered * hyp_centered[:, np.newaxis], axis=0)
            correlations[key_guess, :] = cov / (num_traces * traces_std * hyp_std)
            
    return correlations

def main():
    # PASUL 1: Barem - Benchmarking pe o matrice test
    print(">>> PASUL 1: Validare SVD de la zero")
    test_mat = np.random.rand(100, 50)
    _ = benchmark_svd(test_mat)
    
    # PASUL 2: Încărcare Date
    print(">>> PASUL 2: Încărcare date hardware")
    ASCAD_PATH = "ASCAD.h5"
    traces, plaintexts, real_key = extract_poi(ASCAD_PATH, num_traces=200, time_window=(45000, 45100))
    print(f"Cheia reală (Byte 0): {hex(real_key[0])}")
    
    # PASUL 3: Filtrare Truncated SVD Custom
    print("\n>>> PASUL 3: Filtrarea SVD (Truncated SVD)")
    U, S, Vt = SVD_Custom(traces)
    
    # --- FIX APLICAT: TRUNCATED SVD ---
    # Eliminăm \sigma_0 (ceasul). Păstrăm DOAR indexul 1 (semnalul clar).
    # Nu păstrăm indexurile 2 și 3, ele sunt zgomot termic.
    print("Curățare matrice (Eliminare Macro-Zgomot)...")
    indices_to_keep = [1] 
    
    # Reconstruim A_clean
    A_clean = np.zeros_like(traces)
    for idx in indices_to_keep:
        if idx < len(S):
            A_clean += S[idx] * np.outer(U[:, idx], Vt[idx, :])
            
    # PASUL 4: Atacul CPA
    print("\n>>> PASUL 4: Execuție Atac Pearson (CPA)")
    corr_raw = calculate_cpa(traces, plaintexts)
    corr_clean = calculate_cpa(A_clean, plaintexts)
    
    best_guess_clean = np.argmax(np.max(np.abs(corr_clean), axis=1))
    print(f"Cheia ghicită după filtrarea SVD: {hex(best_guess_clean)}")
    if best_guess_clean == real_key[0]:
        print("[SUCCES] Cheia AES a fost extrasă cu succes!")
    else:
        print("[EȘEC] SVD nu a putut izola leakage-ul (necesită mai multe urme).")

    # PASUL 5: Vizualizare Academică
    print("\n>>> PASUL 5: Generare Raport Grafic")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.set_facecolor("#0a0a0a")
    
    for ax in axes:
        ax.set_facecolor("#0f0f14")
        ax.grid(color="#1a1a2e", linestyle="--", linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1. Comparație Traces
    axes[0].plot(traces[0], color="#ff4455", alpha=0.6, label="Urmă Brută (Ascuns în zgomot)")
    axes[0].plot(A_clean[0], color="#22dd88", linewidth=1.5, label="Urmă Filtrată (Truncated SVD)")
    axes[0].set_title("Efectul SVD asupra datelor brute", color="white")
    axes[0].legend(loc="upper right")

    # 2. Spectrul SVD
    n_display = min(30, len(S))
    colors = ["#b2000f" if i not in indices_to_keep else "#04bd07" for i in range(n_display)]
    axes[1].bar(range(n_display), S[:n_display], color=colors, alpha=0.8)
    axes[1].set_yscale('log')
    axes[1].set_title("Spectrul Valorilor Singulare (Roșu=Tăiat, Verde=Păstrat)", color="white")
    axes[1].set_ylabel("Magnitudine (Log)")

    # 3. CPA pe datele Curățate (A_clean)
    for k in range(256):
        if k == real_key[0]:
            axes[2].plot(np.abs(corr_clean[k]), color="#ff4455", linewidth=2.5, zorder=10, label=f"Cheia Corectă ({hex(k)})")
        else:
            axes[2].plot(np.abs(corr_clean[k]), color="#555566", linewidth=0.5, alpha=0.5)
    
    axes[2].set_title("Rezultatul Atacului CPA (Pearson Correlation pe $A_{clean}$)", color="white")
    axes[2].set_xlabel("Eșantion Timp (T)")
    axes[2].set_ylabel("Corelație Absolută")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("raport_dpa.png", dpi=150, facecolor="#0a0a0a")
    print("Graficul a fost salvat cu succes în 'raport_dpa.png'. Pipeline finalizat.")

if __name__ == "__main__":
    main()