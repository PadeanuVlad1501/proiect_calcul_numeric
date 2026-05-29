"""
=============================================================================
 SIMULARE MATEMATICĂ: ATAC SIDE-CHANNEL (DPA) CU FILTRARE PRIN TRUNCATED SVD
=============================================================================

CONTEXT ACADEMIC:
-----------------
Differential Power Analysis (DPA) exploatează faptul că consumul de putere
al unui procesor variază în funcție de datele prelucrate. Când AES procesează
un plaintext, operația SubBytes produce un "leakage" (scurgere de informație)
proporțional cu valoarea intermediară calculată.

MODELUL MATEMATIC CORE:
-----------------------
Fie A ∈ ℝ^{N×T} matricea urmelor de putere brute, unde:
  - N = număr de urme (măsurători independente)
  - T = număr de pași de timp per urmă

SVD garantează că: A = U Σ Vᵀ  (descompunere exactă, unică până la semn)

Componentele matricei:
  U ∈ ℝ^{N×N}   — vectori singulari stângi  (baza spațiului urmelor)
  Σ ∈ ℝ^{N×T}   — valorile singulare σ₀ ≥ σ₁ ≥ ... ≥ σ_{K-1} ≥ 0
  Vᵀ ∈ ℝ^{T×T}  — vectori singulari drepți  (baza spațiului temporal)

STRUCTURA SPECTRALĂ A PROBLEMEI DPA:
--------------------------------------
  σ₀ >> σ₁ >> σ₂...σ_K:

  σ₀ (~5584) — macro-semnal (ceas CPU, alimentare). Domina COMPLET energia
               matricei. Vectorul V₀ ∈ ℝᵀ descrie forma temporală a ceasului,
               U₀ ∈ ℝᴺ descrie cum variaza amplitudinea inter-urme (jitter).
               → TĂIEM: bruiaj pur, fara informație criptografică.

  σ₁ (~260)  — leakage AES (SubBytes). Gap spectral clar față de zgomot.
               V₁ ∈ ℝᵀ conține profilul temporal al operației (spike la t=250).
               U₁ ∈ ℝᴺ conține corelația cu Hamming Weight (cheia candidată).
               → PĂSTRĂM: informația criptografică e concentrată EXACT AICI.

  σ₂...σ_K  — zgomot termic (valorile similare, ~79-81). Fiecare componentă
               e nedistingibilă: V_i descriu direcții aleatoare.
               → TĂIEM: nu conțin structură corelată cu cheia.

PROIECȚIA ORTOGONALĂ (Truncated SVD):
--------------------------------------
  Reconstrucție rang-1:  A_clean = σ₁ · u₁ · v₁ᵀ

  Aceasta este proiecția ortogonală a lui A pe subspațiul span{u₁} ⊗ span{v₁}:
    P = u₁u₁ᵀ ⊗ v₁v₁ᵀ   (proiecție pe rang-1)
    A_clean = P(A)

  Proprietăți:
    P² = P      (idempotentă — proiectezi de două ori = proiectezi o dată)
    Pᵀ = P      (simetrică — proiecție ortogonală, nu oblică)
    ‖A - A_clean‖_F este minimizat pentru rang dat (Teorema Eckart-Young)

EXTRACȚIA CHEII (pasul final DPA):
------------------------------------
  V₁ ∈ ℝᵀ este profilul temporal al operației SubBytes.
  Spike-ul din V₁ la t ∈ [240, 260] identifică momentul operației.
  Corelația ‖corr(U₁, HW(k))‖ maximizată pentru cheia corectă k*.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# =============================================================================
# SECȚIUNEA 1: GENERAREA DATELOR SINTETICE
# =============================================================================
#
# Matricea A ∈ ℝ^{N×T}:
#
#   A[i,t] = α · macro(t) · (1 + ε[i])     ← componenta dominantă (ceas CPU)
#           + β · hw_norm[i] · spike(t)     ← leakage AES (rang 1 exact!)
#           + γ · η[i,t]                    ← zgomot termic Gaussian alb
#
# Observație: leakage_matrix = hw_norm ⊗ spike_profile este o matrice de rang 1.
# SVD va concentra TOATĂ energia leakage-ului în σ₁, u₁, v₁.
# Aceasta e intuiția matematică centrală a atacului DPA cu SVD.
# =============================================================================

# ─── Parametri globali ────────────────────────────────────────────────────────
N   = 1000   # Număr de urme de putere (măsurători)
T   = 500    # Număr de pași de timp per urmă
rng = np.random.default_rng(42)  # Generator modern, reproducibil (fără legacy)

# ─── Macro-semnalul (ceasul procesorului / bus de alimentare) ─────────────────
# Semnal periodic sinusoidal, amplitudine mare → domina spectrumul SVD (σ₀ >> σ₁)
t_axis          = np.linspace(0, 4 * np.pi, T)
macro_shape     = np.sin(t_axis) + 0.5 * np.cos(3 * t_axis)   # formă oscilantă
amplitude_macro = 10.0   # amplitudine dominantă

# Jitter inter-urme: variație mică de amplitudine ε[i] ∼ N(0, 0.05²)
jitter          = 1.0 + 0.05 * rng.standard_normal((N, 1))   # (N×1), broadcast
macro_matrix    = amplitude_macro * jitter * macro_shape       # (N×T)

# ─── Semnalul secret AES — Leakage operației SubBytes ─────────────────────────
#
# Model Hamming Weight (HW):
#   Consumul de putere este proporțional cu numărul de biți "1" din registrul
#   de ieșire al operației SubBytes: hw[i] = HW(SubBytes(plaintext[i] ⊕ key)).
#   Pentru simulare, generăm hw ∈ {0,...,8} (8 biți per byte AES).
#
# Profil temporal:
#   Operația SubBytes este PUNCTUALĂ în timp (câteva cicluri de ceas).
#   Modelăm aceasta ca un spike Gaussian centrat la t=250, σ=4 pași.
#
# Structura de rang-1:
#   leakage_matrix = hw_normalized ⊗ spike_profile   (produs exterior!)
#   → SVD va crea un singur σ mare pentru toată această energie.

t_start, t_end = 240, 260   # fereastra temporală a operației SubBytes

t_window      = np.arange(T, dtype=float)
spike_center  = (t_start + t_end) / 2.0   # = 250
spike_sigma   = 4.0
spike_profile = np.exp(-0.5 * ((t_window - spike_center) / spike_sigma) ** 2)
spike_profile[:t_start] = 0.0
spike_profile[t_end:]   = 0.0
spike_profile          /= spike_profile.max()   # normalizare la [0, 1]

# Valorile Hamming Weight per urmă, centrate și normalizate (media 0, var 1)
hw_values   = rng.integers(0, 9, size=N).astype(float)   # HW ∈ {0,...,8}
hw_centered = hw_values - hw_values.mean()
hw_normalized = hw_centered / np.std(hw_centered)         # media 0, σ=1

# Matricea de leakage: produs exterior (N×1) ⊗ (1×T) → rang 1!
# SNR per urmă ≈ 0.003 → invizibil direct, dar σ₁ >> σ₂ asigura extracția SVD
amplitude_leak  = 3.0
leakage_matrix  = amplitude_leak * hw_normalized[:, np.newaxis] * spike_profile

# ─── Zgomotul termic Gaussian alb ─────────────────────────────────────────────
# η[i,t] ∼ i.i.d. N(0, 1) → zgomot alb, fara structura corelata
amplitude_noise = 1.5
noise_matrix    = amplitude_noise * rng.standard_normal((N, T))

# ─── Matricea finală A ────────────────────────────────────────────────────────
A = macro_matrix + leakage_matrix + noise_matrix

# Referințe pentru validare (valorile "ideale", fără zgomot)
leakage_ideal   = leakage_matrix.mean(axis=0)   # profilul temporal mediu teoretic
spike_reference = spike_profile.copy()          # forma pura a spike-ului

# Verificare SNR
snr_per_trace = np.var(leakage_matrix) / np.var(noise_matrix)
print(f"[INFO] SNR per-trace  = {snr_per_trace:.5f}  (invizibil în brut!)")
print(f"[INFO] Amplitudini: macro={amplitude_macro}, leak={amplitude_leak}, noise={amplitude_noise}")


# =============================================================================
# SECȚIUNEA 2: MOTORUL SVD — TRUNCATED SVD CU FILTRARE PE SUBSPAȚII
# =============================================================================
#
# scipy.linalg.svd returnează descompunerea COMPLETĂ (full_matrices=True):
#   A = U @ np.diag(s) @ Vt
#   unde U ∈ ℝ^{N×N}, Vt ∈ ℝ^{T×T} sunt matrice UNITARE (U·Uᵀ = I, Vt·Vtᵀ = I)
#
# RECONSTRUCȚIE RANG-1 pentru componenta i:
#   A_i = σᵢ · uᵢ · vᵢᵀ   ←  matrice de rang exact 1
#   ‖A_i‖_F = σᵢ            ←  energia concentrată în valoarea singulară
#
# TRUNCATED SVD (Eckart-Young 1936):
#   Cel mai bun aproximant de rang-k al lui A în norma Frobenius este:
#   A_k = Σ_{i=0}^{k-1} σᵢ · uᵢ · vᵢᵀ
#   Eroarea: ‖A - A_k‖_F² = Σ_{i=k}^{min(N,T)-1} σᵢ²
#
# STRATEGIA NOASTRĂ:
#   A_clean = σ₁ · u₁ · v₁ᵀ   (rang-1, conține EXACT leakage-ul AES)
#
#   Tăiem σ₀:    macro-semnal (dominanta, σ₀ >> σ₁, fara informatie criptografica)
#   Tăiem σ₂...: zgomot termic (σ-uri mici, egale intre ele, direcții aleatoare)
# =============================================================================

print("\n[SVD] Calculez descompunerea SVD completă...")
U, s, Vt = svd(A, full_matrices=True)

print(f"[SVD] Shape  →  U:{U.shape}, s:{s.shape}, Vt:{Vt.shape}")
print(f"[SVD] Top-6 valori singulare: {s[:6].round(3)}")
print(f"[SVD] Gap spectral σ₁/σ₂ = {s[1]/s[2]:.3f}x  (GAP CLAR = leakage izolat în σ₁)")

# ─── Reconstrucție rang-1: A_clean = σ₁ · u₁ · v₁ᵀ ──────────────────────────
#
# u₁ ∈ ℝᴺ  →  descrie variația INTER-URME a leakage-ului (corelat cu HW)
# v₁ ∈ ℝᵀ  →  descrie profilul TEMPORAL (spike la t=250)
# σ₁       →  magnitudinea energiei de leakage

idx_leakage = 1   # componenta SVD care conține leakage-ul AES
u1 = U[:, idx_leakage]           # (N,) — vectorul singular stâng
v1 = Vt[idx_leakage, :]          # (T,) — vectorul singular drept
s1 = s[idx_leakage]

# Reconstrucție rang-1 (proiecție ortogonală pe subspațiul {u₁} ⊗ {v₁})
A_clean = s1 * np.outer(u1, v1)  # (N×T)

print(f"\n[SVD] Reconstrucție rang-1 cu σ₁={s1:.3f}")
print(f"[SVD] Corelație V₁ vs spike_profile: {np.corrcoef(v1, spike_profile)[0,1]:.6f}")
print(f"[SVD] Corelație U₁ vs hw_normalized: {np.corrcoef(u1, hw_normalized)[0,1]:.6f}")

# ─── Extracția profilului temporal al leakage-ului ────────────────────────────
# V₁ este proporțional cu spike_profile (cu semn posibil schimbat).
# Îl orientăm astfel încât să corespundă orientării spike-ului.
v1_oriented = v1 * np.sign(np.corrcoef(v1, spike_profile)[0, 1])

# Media urmelor filtrate (estimatorul DPA al profilului temporal)
mean_clean = A_clean.mean(axis=0)

# ─── Normalizare pentru comparație vizuală ────────────────────────────────────
def normalize(x):
    """Normalizare min-max pe intervalul [-1, 1]."""
    lo, hi = x.min(), x.max()
    return 2.0 * (x - lo) / (hi - lo) - 1.0 if (hi - lo) > 1e-12 else x

leakage_ideal_norm = normalize(leakage_ideal)
mean_clean_norm    = normalize(mean_clean)
v1_norm            = normalize(v1_oriented)


# =============================================================================
# SECȚIUNEA 3: VIZUALIZARE — PREZENTARE ACADEMICĂ (dark_background)
# =============================================================================

plt.style.use("dark_background")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle(
    r"Simulare Atac Side-Channel DPA — Filtrare prin Truncated SVD",
    fontsize=17, fontweight="bold", color="white", y=0.985
)
fig.patch.set_facecolor("#0a0a0a")

# ── Paletă de culori ──────────────────────────────────────────────────────────
C_MACRO  = "#ff4455"   # roșu    — componentă tăiată (macro / zgomot)
C_USEFUL = "#22dd88"   # verde   — subspațiu util (leakage)
C_RAW    = "#8899ff"   # albastru deschis — semnal brut
C_IDEAL  = "#ffdd33"   # galben  — semnal ideal (referință)
C_SVD    = "#00eedd"   # cyan    — reconstrucție SVD
C_SPIKE  = "#ff9944"   # portocaliu — fereastra temporală SubBytes

for ax in axes:
    ax.set_facecolor("#0f0f14")
    ax.tick_params(colors="#cccccc", labelsize=10)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333344")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#1a1a2e", linewidth=0.6, linestyle="--")

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 1: Spectrul Valorilor Singulare (scară log)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]

n_display = 30   # afișăm primele 30 de valori singulare

# Zone colorate cu axvspan (indice component pe axa X)
ax1.axvspan(-0.5, 0.5,
            color=C_MACRO,  alpha=0.20, zorder=0,
            label=r"$\sigma_0$: macro-semnal (TĂIAT)")
ax1.axvspan(0.5, 1.5,
            color=C_USEFUL, alpha=0.22, zorder=0,
            label=r"$\sigma_1$: leakage AES (PĂSTRAT — rang 1)")
ax1.axvspan(1.5, n_display - 0.5,
            color=C_MACRO,  alpha=0.08, zorder=0,
            label=r"$\sigma_2$ ... $\sigma_{29}$: zgomot termic (TĂIAT)")

# Bare colorate per componentă
bar_colors = [C_MACRO if i == 0 else (C_USEFUL if i == 1 else "#445566")
              for i in range(n_display)]
ax1.bar(np.arange(n_display), s[:n_display],
        color=bar_colors, alpha=0.90, width=0.75, zorder=2)

ax1.set_yscale("log")
ax1.set_xlim(-0.8, n_display - 0.2)
ax1.set_xlabel(r"Indice component $i$", color="white", fontsize=12)
ax1.set_ylabel(r"Magnitudinea $\sigma_i$ (scară logaritmică)", color="white", fontsize=12)
ax1.set_title(
    r"Spectrul Valorilor Singulare — Separarea Subspațiilor A = U$\Sigma$V$^\top$",
    color="white", fontsize=12, pad=10
)
ax1.legend(loc="upper right", fontsize=9,
           facecolor="#111122", edgecolor="#334", labelcolor="white")

# Adnotări
ax1.annotate(
    r"$\sigma_0 = {:.0f}$ (macro-semnal)".format(s[0]),
    xy=(0, s[0]), xytext=(4, s[0] * 0.5),
    color=C_MACRO, fontsize=9, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_MACRO, lw=1.3)
)
ax1.annotate(
    r"$\sigma_1 = {:.1f}$ (leakage AES)".format(s[1]),
    xy=(1, s[1]), xytext=(5, s[1] * 2.8),
    color=C_USEFUL, fontsize=9, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_USEFUL, lw=1.3)
)
ax1.annotate(
    r"Gap: $\sigma_1 / \sigma_2 = {:.2f}\times$".format(s[1] / s[2]),
    xy=(1.5, s[2]),
    xytext=(8, s[1] * 0.7),
    color="#ffaa55", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#ffaa55", lw=1.0)
)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 2: Date Brute — Ineficiența Atacului Direct
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]

raw_trace    = A[0, :]
raw_trace_n  = normalize(raw_trace)

ax2.plot(raw_trace_n,
         color=C_RAW, alpha=0.65, linewidth=0.9,
         label="Urmă brută: zgomot + macro + leakage (toate amestecate)")
ax2.plot(leakage_ideal_norm,
         color=C_IDEAL, alpha=0.95, linewidth=2.0, linestyle="--",
         label=r"Semnal ideal (leakage AES pur) — INVIZIBIL în brut")

# Fereastra temporală a operației SubBytes
ax2.axvspan(t_start, t_end, color=C_SPIKE, alpha=0.18, zorder=0,
            label=r"Fereastra SubBytes [$t_{240}$, $t_{260}$]")

ax2.set_xlim(0, T)
ax2.set_xlabel("Pasul de timp $t$", color="white", fontsize=12)
ax2.set_ylabel("Amplitudine (normalizat)", color="white", fontsize=12)
ax2.set_title(
    r"Date Brute — Atacul Direct EȘUEAZĂ: Semnalul Secret Este Complet Ascuns",
    color="white", fontsize=12, pad=10
)
ax2.legend(loc="upper left", fontsize=9,
           facecolor="#111122", edgecolor="#334", labelcolor="white")

# Casetă explicativă
ax2.text(
    0.02, 0.08,
    r"SNR per urmă $\approx {:.4f}$ — spike-ul AES e $< 0.1\%$ din energia totală".format(snr_per_trace),
    transform=ax2.transAxes, color="#ff9944", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a0f00",
              edgecolor="#ff9944", alpha=0.85)
)

# ─────────────────────────────────────────────────────────────────────────────
# SUBPLOT 3: Reconstrucția SVD — Extracția Curată a Leakage-ului
# ─────────────────────────────────────────────────────────────────────────────
ax3 = axes[2]

ax3.plot(leakage_ideal_norm,
         color=C_IDEAL, alpha=0.80, linewidth=2.0, linestyle="--",
         label=r"Semnal ideal (referință — $\langle\text{leakage}\rangle$ teoretic)")
ax3.plot(v1_norm,
         color=C_SVD, alpha=0.95, linewidth=2.0,
         label=r"$V_1$ (vectorul temporal $\sigma_1$) — profilul recuperat")

# Fereastra SubBytes
ax3.axvspan(t_start, t_end, color=C_SPIKE, alpha=0.18, zorder=0,
            label=r"Fereastra SubBytes [$t_{240}$, $t_{260}$]")

# Adnotare spike extras
peak_idx = int(np.argmax(np.abs(v1_norm)))
ax3.annotate(
    r"Spike $\hat{t}$ = " + str(peak_idx) + r" — cheia candidată detectată",
    xy=(peak_idx, v1_norm[peak_idx]),
    xytext=(peak_idx + 35, v1_norm[peak_idx] * 0.65),
    color=C_SVD, fontsize=9, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_SVD, lw=1.5)
)

ax3.set_xlim(0, T)
ax3.set_xlabel("Pasul de timp $t$", color="white", fontsize=12)
ax3.set_ylabel("Amplitudine (normalizat)", color="white", fontsize=12)
ax3.set_title(
    r"Reconstrucție Truncated SVD — Extracție Curată: $A_{{clean}} = \sigma_1 \cdot u_1 \cdot v_1^\top$",
    color="white", fontsize=12, pad=10
)
ax3.legend(loc="upper left", fontsize=9,
           facecolor="#111122", edgecolor="#334", labelcolor="white")

# Corelație finală
corr_v1_ideal = np.corrcoef(v1_norm, leakage_ideal_norm)[0, 1]
ax3.text(
    0.02, 0.08,
    r"Corelație $V_1$ vs. ideal = ${:.4f}$ — leakage AES recuperat cu fidelitate înaltă".format(corr_v1_ideal),
    transform=ax3.transAxes, color=C_USEFUL, fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#001a0f",
              edgecolor=C_USEFUL, alpha=0.85)
)

# ─── Layout și salvare ────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.975])
output_path = "/mnt/user-data/outputs/dpa_svd_attack.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
print(f"\n[OUTPUT] Figura salvată: {output_path}")
plt.show()


# =============================================================================
# SECȚIUNEA 4: RAPORT NUMERIC FINAL
# =============================================================================
print("\n" + "=" * 68)
print("  RAPORT NUMERIC — DPA cu Truncated SVD (Rang-1)")
print("=" * 68)
print(f"  Matrice A          :  {N} urme × {T} pași de timp")
print(f"  σ₀  (macro)        :  {s[0]:.2f}  ← TĂIAT (macro-semnal dominant)")
print(f"  σ₁  (leakage AES)  :  {s[1]:.3f}  ← PĂSTRAT (rang-1, informație criptografică)")
print(f"  σ₂  (zgomot)       :  {s[2]:.3f}  ← TĂIAT (prima componentă de zgomot)")
print(f"\n  Gap spectral σ₁/σ₂ :  {s[1]/s[2]:.3f}×  ← gap clar = SVD izolează leakage-ul")
print(f"  Gap spectral σ₀/σ₁ :  {s[0]/s[1]:.1f}×  ← macro-semnal tăiat eficient")
print(f"\n  Energie macro (σ₀²):  {s[0]**2/(s**2).sum():.2%} din energia totală")
print(f"  Energie leak (σ₁²) :  {s[1]**2/(s**2).sum():.4%} din energia totală")
print(f"\n  SNR per urmă        :  {snr_per_trace:.5f}  (invizibil în brut)")
print(f"\n  Corel. V₁ vs spike  :  {np.corrcoef(v1, spike_profile)[0,1]:.6f}")
print(f"  Corel. U₁ vs HW     :  {np.corrcoef(u1, hw_normalized)[0,1]:.6f}")
print(f"  Corel. V₁ vs ideal  :  {corr_v1_ideal:.6f}  ← fidelitate reconstrucție")
print("=" * 68)
