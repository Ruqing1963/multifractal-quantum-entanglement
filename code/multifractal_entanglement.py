"""
Multifractal singularity spectrum f(alpha) <-> Renyi entanglement spectrum.

Input: a two-scale binomial Cantor measure (p1, p2; r1, r2).
Output: tau(q), D_q, f(alpha), annotated with the entanglement quantities they become when the
measure is read as the eigenvalue distribution of a reduced density matrix rho_A at resolution eps:

    ln Tr rho_A^q = -tau(q) L,           L = ln(1/eps)
    S_q           = D_q L                 (Renyi entropy, D_q = tau(q)/(q-1))
    S~_q          = q^2 d/dq[(q-1)/q S_q] = f(alpha_q) L     (refined Renyi = escort entropy)
    <K>_q         = alpha_q L             (modular energy in the escort state)
    D_0  -> Hartley entropy  ln rank(rho_A) / L
    D_1  -> von Neumann entropy / L
    D_2  -> collision entropy  -ln Tr rho_A^2 / L
    D_oo = alpha_min -> min-entropy  -ln lambda_max / L

Checks performed numerically:
  (1) exact eigenvalue list of the cascade at finite eps  ->  S_q / L  vs  D_q,  S~_q / L  vs  f(alpha_q)
  (2) equal-scale case r = (1/2, 1/2): rho_A = tensor product of n "bond" qubits with Schmidt
      weights (p1, p2) (a random-tensor-network min cut); then D_q = H_q(p)/ln 2 exactly.
  (3) comparison with the quantum-Hall wave-function spectrum (Evers-Mildenberger-Mirlin 2008):
      Delta_q = 2q(1-q)[b0 + b1 (q-1/2)^2], b0 = 0.1291, b1 = 0.0029, d = 2,
      and the exact symmetry f(2d - alpha) = f(alpha) + d - alpha (Mirlin et al. 2006).
      NOTE: this is the multifractality of |psi(r)|^2 (a participation spectrum), not of rho_A.

Usage:  python multifractal_entanglement.py [--p 0.6 0.4] [--r 0.25 0.4] [--show]

Figures are written to ./figures (PNG and PDF), data to ./data (CSV); override with the environment
variables MQE_FIG_DIR and MQE_DATA_DIR.

Companion code for: R. Chen, "Multifractal Large Deviations in Quantum Entanglement" (2026),
DOI: 10.5281/zenodo.23193512
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gammaln


FIG_DIR = os.environ.get("MQE_FIG_DIR", "figures")
DATA_DIR = os.environ.get("MQE_DATA_DIR", "data")


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=150)


def savedata(name, columns, cols, meta=()):
    os.makedirs(DATA_DIR, exist_ok=True)
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    with open(os.path.join(DATA_DIR, f"{name}.csv"), "w", encoding="utf-8", newline="\n") as fh:
        for line in meta:
            fh.write(f"# {line}\n")
        fh.write(",".join(columns) + "\n")
        np.savetxt(fh, arr, delimiter=",", fmt="%.10g")


def logsumexp(v):
    v = np.asarray(v, dtype=float)
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


# ------------------------------------------------------------------ multifractal formalism
def tau_alpha_f(q, p, r):
    """Solve sum_i p_i^q r_i^(-tau) = 1; return tau, alpha = tau'(q), f = q alpha - tau."""
    lp, lr = np.log(p), np.log(r)
    tau = brentq(lambda t: logsumexp(q * lp - t * lr), -1e3, 1e3)
    w = np.exp(q * lp - tau * lr)                     # escort weights, sum = 1
    alpha = np.sum(w * lp) / np.sum(w * lr)
    return tau, alpha, q * alpha - tau


def spectrum(p, r, qs):
    T = np.array([tau_alpha_f(q, p, r) for q in qs])
    tau, alpha, f = T.T
    Dq = np.where(np.abs(qs - 1) > 1e-9, tau / np.where(np.abs(qs - 1) > 1e-9, qs - 1, 1), alpha)
    return tau, Dq, alpha, f


# ------------------------------------------------------------------ exact "entanglement spectrum"
def cascade_eigenvalues(p, r, eps):
    """
    Masses of the cascade at resolution eps (stopping rule: first prefix of size <= eps), grouped by
    composition (k1, k2). Returns (ln eigenvalue, ln multiplicity). They sum to 1 exactly.
    """
    lp, lr, le = np.log(p), np.log(r), np.log(eps)
    lam, mult = [], []
    for k1 in range(int(le / lr[0]) + 2):
        for k2 in range(int(le / lr[1]) + 2):
            if k1 * lr[0] + k2 * lr[1] > le:
                continue
            terms = []
            if k1 >= 1 and (k1 - 1) * lr[0] + k2 * lr[1] > le:
                terms.append(gammaln(k1 + k2) - gammaln(k1) - gammaln(k2 + 1))
            if k2 >= 1 and k1 * lr[0] + (k2 - 1) * lr[1] > le:
                terms.append(gammaln(k1 + k2) - gammaln(k1 + 1) - gammaln(k2))
            if terms:
                lam.append(k1 * lp[0] + k2 * lp[1])
                mult.append(logsumexp(terms))
    return np.array(lam), np.array(mult)


def renyi(q, lam, mult):
    """Renyi entropy of the eigenvalue list (q = 1: von Neumann; q = 0: Hartley)."""
    if abs(q - 1) < 1e-12:
        return -np.sum(np.exp(mult + lam) * lam)
    if q == 0:
        return logsumexp(mult)
    return logsumexp(mult + q * lam) / (1 - q)


def refined_renyi(q, lam, mult):
    """S~_q = von Neumann entropy of the escort state rho^q / Tr rho^q."""
    lw = mult + q * lam
    lw = lw - logsumexp(lw)
    return -np.sum(np.exp(lw) * (lw - mult))


# ------------------------------------------------------------------ quantum-Hall comparison
def qhe_spectrum(qs, b0=0.1291, b1=0.0029, d=2):
    Delta = 2 * qs * (1 - qs) * (b0 + b1 * (qs - 0.5) ** 2)
    tau = d * (qs - 1) + Delta
    alpha = np.gradient(tau, qs)
    return alpha, qs * alpha - tau


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--p", type=float, nargs=2, default=[0.6, 0.4], help="weights p1 p2 (sum to 1)")
    ap.add_argument("--r", type=float, nargs=2, default=[0.25, 0.4], help="scale ratios r1 r2 (sum <= 1)")
    ap.add_argument("--eps", type=float, default=1e-80, help="resolution for the exact spectrum")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    p, r = np.array(a.p), np.array(a.r)
    if not np.isclose(p.sum(), 1):
        ap.error("p1 + p2 must equal 1")
    L = np.log(1 / a.eps)

    qs = np.linspace(-15, 15, 1201)
    tau, Dq, alpha, f = spectrum(p, r, qs)
    D0 = -tau_alpha_f(0, p, r)[0]
    D1 = tau_alpha_f(1, p, r)[1]
    D2 = tau_alpha_f(2, p, r)[0]
    a_min, a_max = tau_alpha_f(200, p, r)[1], tau_alpha_f(-200, p, r)[1]

    print(f"two-scale Cantor measure: p = {tuple(p)}, r = {tuple(r)}")
    print(f"  D_0 (Hartley)      = {D0:.6f}")
    print(f"  D_1 (von Neumann)  = {D1:.6f}")
    print(f"  D_2 (collision)    = {D2:.6f}")
    print(f"  alpha_min = D_inf  = {a_min:.6f}   (min-entropy)")
    print(f"  alpha_max          = {a_max:.6f}   (smallest eigenvalues)")

    # (1) exact eigenvalue list
    lam, mult = cascade_eigenvalues(p, r, a.eps)
    print(f"\n(1) exact cascade spectrum at eps = {a.eps:g}: {len(lam)} distinct eigenvalues, "
          f"trace = {np.exp(logsumexp(mult + lam)):.12f}")
    print("     q      D_q     S_q/L(exact)    f(alpha_q)   S~_q/L(exact)")
    chk = []
    for q in [0, 0.5, 1, 2, 3, 5]:
        t, al, fq = tau_alpha_f(q, p, r)
        dq = al if q == 1 else t / (q - 1)
        sq, st = renyi(q, lam, mult) / L, refined_renyi(q, lam, mult) / L
        chk.append([q, dq, sq, fq, st])
        print(f"   {q:4.1f}  {dq:8.5f}   {sq:8.5f}      {fq:8.5f}    {st:8.5f}")
    meta = [f"two-scale Cantor measure p={tuple(p)}, r={tuple(r)}"]
    savedata("cantor_tau_D_alpha_f", ["q", "tau", "D_q", "alpha", "f"], [qs, tau, Dq, alpha, f], meta)
    C = np.array(chk)
    savedata("cantor_exact_spectrum_check", ["q", "D_q", "S_q_over_L_exact", "f_alpha_q", "S_refined_over_L_exact"],
             [C[:, i] for i in range(5)], meta + [f"exact eigenvalue list at eps={a.eps:g}, L=ln(1/eps)"])

    # (2) tensor product of n bond qubits, r = (1/2, 1/2)
    n = 40
    print(f"\n(2) rho_A = tensor product of n = {n} bond qubits with Schmidt weights {tuple(p)}:")
    for q in [0.5, 2, 3]:
        Hq = np.log(np.sum(p ** q)) / (1 - q)
        tq = tau_alpha_f(q, p, np.array([0.5, 0.5]))[0]
        print(f"   q = {q}:  S_q / (n ln2) = {Hq / np.log(2):.10f}   D_q = {tq / (q - 1):.10f}")

    # (3) quantum-Hall wave-function spectrum
    qq = np.linspace(-2.5, 3.5, 1201)
    aq, fq = qhe_spectrum(qq)
    sym = np.interp(4 - aq, aq[::-1], fq[::-1]) - (fq + 2 - aq)
    ok = (4 - aq > aq.min()) & (4 - aq < aq.max())
    print(f"\n(3) quantum-Hall spectrum: alpha_0 = {aq[np.argmin(np.abs(qq))]:.4f}, "
          f"max |f(2d-alpha) - f(alpha) - d + alpha| = {np.max(np.abs(sym[ok])):.1e}")

    # ------------------------------------------------------------------ figures
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    ax[0, 0].plot(qs, tau, "k")
    ax[0, 0].axhline(0, color="grey", lw=0.5); ax[0, 0].axvline(1, color="grey", lw=0.5)
    ax[0, 0].plot([0, 1], [-D0, 0], "o", color="tab:red")
    ax[0, 0].annotate("tau(0) = -D_0\n(log rank of rho_A)", (0, -D0), (-12, 2), arrowprops=dict(arrowstyle="->"))
    ax[0, 0].annotate("tau(1) = 0\n(Tr rho_A = 1)", (1, 0), (4, -6), arrowprops=dict(arrowstyle="->"))
    ax[0, 0].set_xlabel("q"); ax[0, 0].set_ylabel("tau(q)")
    ax[0, 0].set_title("Mass exponent:  ln Tr rho_A^q = -tau(q) ln(1/eps)")

    ax[0, 1].plot(qs, Dq, "k")
    for q0, val, lab in [(0, D0, "D_0: Hartley"), (1, D1, "D_1: von Neumann"), (2, D2, "D_2: collision")]:
        ax[0, 1].plot(q0, val, "o", color="tab:red")
        ax[0, 1].annotate(lab, (q0, val), (q0 + 2, val + 0.03 * (3 - q0)), arrowprops=dict(arrowstyle="->"))
    ax[0, 1].axhline(a_min, ls=":", color="tab:blue"); ax[0, 1].axhline(a_max, ls=":", color="tab:blue")
    ax[0, 1].text(1, a_min + 0.01, "D_inf = alpha_min: min-entropy", color="tab:blue")
    ax[0, 1].text(-14, a_max - 0.05, "D_-inf = alpha_max", color="tab:blue")
    ax[0, 1].set_xlabel("q"); ax[0, 1].set_ylabel("D_q")
    ax[0, 1].set_title("Generalised dimensions = Renyi entropies per ln(1/eps)")

    al_i = -lam / L
    bins = np.linspace(al_i.min(), al_i.max(), 90)
    idx = np.digitize(al_i, bins)
    a_h = [np.mean(al_i[idx == j]) for j in np.unique(idx)]
    f_h = [logsumexp(mult[idx == j]) / L for j in np.unique(idx)]
    ax[1, 0].plot(alpha, f, "k", lw=2, label="Legendre transform f = q alpha - tau")
    ax[1, 0].plot(a_h, f_h, ".", color="tab:red", ms=4, label=f"eigenvalue histogram, eps = {a.eps:g}")
    ax[1, 0].plot([0, a_max], [0, a_max], ":", color="grey", label="f = alpha")
    ax[1, 0].plot(D1, D1, "o", color="tab:green")
    ax[1, 0].annotate("q = 1: f = alpha = D_1\n(von Neumann; tangent to f = alpha)", (D1, D1),
                      (D1 + 0.05, D1 - 0.25), arrowprops=dict(arrowstyle="->"))
    t0, al0, f0 = tau_alpha_f(0, p, r)
    ax[1, 0].plot(al0, f0, "o", color="tab:purple")
    ax[1, 0].annotate("q = 0: f_max = D_0 (Hartley)", (al0, f0), (al0 + 0.05, f0 + 0.05),
                      arrowprops=dict(arrowstyle="->"))
    ax[1, 0].set_xlim(a_min - 0.05, a_max + 0.05); ax[1, 0].set_ylim(-0.02, D0 + 0.2)
    ax[1, 0].set_xlabel("alpha = -ln(lambda) / ln(1/eps)")
    ax[1, 0].set_ylabel("f(alpha) = ln #(eigenvalues) / ln(1/eps)")
    ax[1, 0].set_title("Singularity spectrum = entanglement-spectrum density\nf(alpha_q) = refined Renyi S~_q / ln(1/eps)")
    ax[1, 0].legend(fontsize=8, loc="lower center")

    ax[1, 1].plot(aq, fq, "k", lw=2, label="quantum Hall, Delta_q = 2q(1-q)[b0 + b1(q-1/2)^2]")
    gam = 2 * 0.1291
    aa = np.linspace(aq.min(), aq.max(), 400)
    ax[1, 1].plot(aa, 2 - (aa - 2 - gam) ** 2 / (4 * gam), "--", color="tab:orange",
                  label=f"parabolic, gamma = 2 b0 = {gam:.4f}")
    ax[1, 1].axhline(0, color="grey", lw=0.5)
    ax[1, 1].set_xlabel("alpha"); ax[1, 1].set_ylabel("f(alpha)")
    ax[1, 1].set_title("Wave-function multifractality at the quantum Hall transition\n"
                       "(participation spectrum of |psi|^2, not of rho_A)")
    ax[1, 1].legend(fontsize=8)

    savefig(fig, "multifractal_entanglement")
    savedata("cantor_eigenvalue_histogram", ["alpha", "f_hist"], [a_h, f_h], meta + [f"eps={a.eps:g}"])
    savedata("quantum_hall_spectrum", ["q", "alpha", "f"], [qq, aq, fq],
             ["Delta_q = 2q(1-q)[b0 + b1 (q-1/2)^2], b0=0.1291, b1=0.0029, d=2 (Evers-Mildenberger-Mirlin 2008)"])
    print(f"\nfigures written to {FIG_DIR}/, data to {DATA_DIR}/")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
