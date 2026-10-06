"""
Frontier-2 deep dive: numerical checks for four questions.

  1. Free fermions (Peschel): ln Tr rho_A^q = sum_k ln(zeta_k^q + (1-zeta_k)^q).
     Exact consequence: the refined Renyi entropy is the entropy of an ideal Fermi gas with the
     entanglement energies eps_k at temperature 1/q,  S~_q = sum_k s_F(1/(1+e^{q eps_k})).
     Example: critical power-law random banded matrix (PRBM, unitary class, b = 1):
     wave-function spectrum f_wave vs many-body entanglement spectrum f_ent.
  2. Markov-modulated cascade: tau(q) = -ln Lambda_q / ln b with Lambda_q the Perron root of T D_q.
     First order: Delta tau = -(lambda_2/ln b) Var_pi(z_q)/<z_q>^2. Analytic in q for lambda_2 < 1.
  3. Quantum-Hall Delta_q in the Weyl-invariant ("Casimir") variable u = q(1-q).
  4. Gauss-Bonnet cosmic branes in d = 4: x_q, S_q, f(alpha_q); t_2 ratio vs (1-2a)/(1-6a);
     sign change of the refined Renyi entropy at finite q for a > 0.

Usage:  python frontier2_deepdive.py [--show]

Figures are written to ./figures (PNG and PDF), data to ./data (CSV); override with the environment
variables MQE_FIG_DIR and MQE_DATA_DIR.

Companion code for: R. Chen, "Multifractal Large Deviations in Quantum Entanglement" (2026),
DOI: 10.5281/zenodo.23193512
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

rng = np.random.default_rng(20261006)
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


def sF(n):
    n = np.clip(n, 1e-300, 1 - 1e-16)
    return -n * np.log(n) - (1 - n) * np.log1p(-n)


# =============================================================== 1. free fermions on a PRBM
def prbm(N, b):
    i = np.arange(N)
    r = np.abs(i[:, None] - i[None, :])
    r = np.minimum(r, N - r)                                  # periodic distance
    sig = 1.0 / np.sqrt(1.0 + (r / b) ** 2)
    A = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
    H = A * sig
    H = (H + H.conj().T) / np.sqrt(2)
    return H


def part1(N=2048, b=1.0, nreal=6):
    qs = np.linspace(0.25, 8, 300)
    G = np.zeros_like(qs); Ssum = 0.0
    eps_all = []
    lsizes = 2 ** np.arange(1, int(np.log2(N)) - 2)
    qw = np.linspace(-1.5, 2.5, 161)
    Pq = np.zeros((len(qw), len(lsizes)))
    fermi_dev = 0.0
    for _ in range(nreal):
        E, V = np.linalg.eigh(prbm(N, b))
        occ = V[:, :N // 2]                                      # half filling
        C = occ[:N // 2].conj() @ occ[:N // 2].T                 # C_ij, i,j in A = first half
        zeta = np.clip(np.linalg.eigvalsh(C), 1e-15, 1 - 1e-15)
        eps = np.log((1 - zeta) / zeta)
        eps_all.append(eps)
        G += np.array([np.sum(np.log(zeta ** q + (1 - zeta) ** q)) for q in qs])
        Ssum += np.sum(sF(zeta))
        # exact Fermi-gas identity for the refined Renyi entropy, checked at q = 2.5
        q0 = 2.5
        lnZq = np.sum(np.log(zeta ** q0 + (1 - zeta) ** q0))
        h = 1e-5
        Gp = (np.sum(np.log(zeta ** (q0 + h) + (1 - zeta) ** (q0 + h)))
              - np.sum(np.log(zeta ** (q0 - h) + (1 - zeta) ** (q0 - h)))) / (2 * h)
        St_legendre = lnZq - q0 * Gp
        St_fermi = np.sum(sF(1 / (1 + np.exp(np.clip(q0 * eps, -700, 700)))))
        fermi_dev = max(fermi_dev, abs(St_legendre - St_fermi) / St_fermi)
        # wave-function multifractality of states near the band centre
        sel = np.argsort(np.abs(E))[:64]
        psi2 = np.abs(V[:, sel]) ** 2
        for jl, l in enumerate(lsizes):
            mu = psi2.reshape(N // l, l, -1).sum(axis=1)
            for iq, q in enumerate(qw):
                Pq[iq, jl] += np.mean(np.sum(mu ** q, axis=0))
    G /= nreal; S1 = Ssum / nreal
    # entanglement spectrum, normalised by S_1  (L := S_1)
    tau_e = -G / S1
    al_e = np.gradient(tau_e, qs); f_e = qs * al_e - tau_e
    # wave-function spectrum (annealed average over states and realisations)
    tau_w = np.array([np.polyfit(np.log(lsizes / N), np.log(Pq[iq] / nreal), 1)[0] for iq in range(len(qw))])
    al_w = np.gradient(tau_w, qw); f_w = qw * al_w - tau_w
    print(f"[1] PRBM N={N}, b={b}, {nreal} realisations: S_1 = {S1:.3f}")
    print(f"    Fermi-gas identity S~_q = sum s_F(1/(1+e^(q eps))): max rel. deviation = {fermi_dev:.1e}")
    print(f"    wave function: D_2 = {tau_w[np.argmin(np.abs(qw - 2))]:.3f}, alpha_0 = {al_w[np.argmin(np.abs(qw))]:.3f}, "
          f"min f_wave = {f_w.min():.3f}")
    print(f"    entanglement:  min f_ent/S_1 = {f_e.min():.3f} (>= 0 always)")
    meta = [f"PRBM unitary class, N={N}, b={b}, {nreal} realisations, half filling, A = half chain"]
    savedata("prbm_wavefunction_spectrum", ["q", "tau", "alpha", "f"], [qw, tau_w, al_w, f_w],
             meta + ["annealed average over the 64 states closest to the band centre"])
    savedata("prbm_entanglement_spectrum", ["q", "tau_over_S1", "alpha_over_S1", "f_over_S1"], [qs, tau_e, al_e, f_e],
             meta + [f"S_1 = {S1:.6f}; Fermi-gas identity max rel. deviation = {fermi_dev:.2e}"])
    return qs, al_e, f_e, al_w, f_w, np.concatenate(eps_all)


# =============================================================== 2. Markov-modulated cascade
def part2():
    P = [np.array([0.7, 0.3]), np.array([0.9, 0.1])]           # bond Schmidt spectra of two sectors
    lb = np.log(2)
    qs = np.linspace(-6, 10, 801)
    z = np.array([[np.sum(p ** q) for p in P] for q in qs])    # z_q(s)

    def tau_exact(l2):
        e = (1 - l2) / 2                                       # symmetric 2-state chain, pi = (1/2, 1/2)
        T = np.array([[1 - e, e], [e, 1 - e]])
        return np.array([-np.log(np.max(np.abs(np.linalg.eigvals(T @ np.diag(zz))))) / lb for zz in z])

    tau0 = -np.log(z.mean(axis=1)) / lb
    var = z.var(axis=1)
    print("[2] Markov cascade: first-order Delta tau = -(lambda_2/ln b) Var(z_q)/<z_q>^2")
    for l2 in (0.05, 0.2):
        dt_ex = tau_exact(l2) - tau0
        dt_pt = -(l2 / lb) * var / z.mean(axis=1) ** 2
        sel = np.abs(qs) < 4
        print(f"    lambda_2 = {l2}: max |exact - first order| / max|exact| = "
              f"{np.max(np.abs(dt_ex - dt_pt)[sel]) / np.max(np.abs(dt_ex[sel])):.2e}")
    # finite-n transfer-matrix check of the Perron-root formula
    l2 = 0.5; e = (1 - l2) / 2; T = np.array([[1 - e, e], [e, 1 - e]]); n = 400
    q = 2.5; D = np.diag([np.sum(p ** q) for p in P]); v = np.array([0.5, 0.5]) @ D; lnZ = 0.0
    for _ in range(n - 1):
        v = v @ T @ D; s = v.sum(); lnZ += np.log(s); v /= s
    lnZ += np.log(v.sum())
    print(f"    transfer matrix n={n}, q={q}, lambda_2={l2}: -lnZ/(n ln2) = {-lnZ / (n * lb):.6f}, "
          f"Perron formula = {tau_exact(l2)[np.argmin(np.abs(qs - q))]:.6f}")
    out = {}
    cols, names = [qs], ["q"]
    for l2 in (0.0, 0.9, 0.99, 1.0):
        t = -np.log(z.max(axis=1)) / lb if l2 == 1.0 else tau_exact(l2)
        a = np.gradient(t, qs)
        out[l2] = (a, qs * a - t)
        cols += [t, a, qs * a - t]; names += [f"tau_l{l2}", f"alpha_l{l2}", f"f_l{l2}"]
    savedata("markov_cascade", names, cols,
             ["two-state symmetric Markov cascade, bond spectra (0.7,0.3) and (0.9,0.1), b=2, annealed over sectors"])
    return out


# =============================================================== 3. quantum-Hall Delta_q in u = q(1-q)
def part3(b0=0.1291, b1=0.0029):
    c1, c2 = 2 * (b0 + b1 / 4), -2 * b1
    q = np.linspace(-1, 2, 7)
    u = q * (1 - q)
    lhs = 2 * u * (b0 + b1 * (q - 0.5) ** 2)
    print(f"[3] Delta_q = c1 u + c2 u^2 with c1 = {c1:.4f}, c2 = {c2:.4f};  "
          f"max |EMM form - Casimir form| = {np.max(np.abs(lhs - (c1 * u + c2 * u ** 2))):.1e};  "
          f"alpha_0 - 2 = c1 = {c1:.4f}")


# =============================================================== 4. Gauss-Bonnet cosmic branes, d = 4
def gb_spectrum(a, d=4, qs=None):
    """a = lambda_GB f_inf. T(x)/T_0 and Jacobson-Myers horizon entropy S(x)/S(1) for hyperbolic GB black holes."""
    T = lambda x: (d * (1 - a) * x ** 4 - (d - 2) * x ** 2 + (d - 4) * a) / (2 * x * (x ** 2 - 2 * a))
    S = lambda x: x ** (d - 1) * (1 - 2 * a * (d - 1) / ((d - 3) * x ** 2)) / (1 - 2 * a * (d - 1) / (d - 3))
    x0 = brentq(lambda x: d * (1 - a) * x ** 4 - (d - 2) * x ** 2 + (d - 4) * a, np.sqrt(max(2 * a, 0.0)) + 1e-9, 1.0)
    xq = np.array([brentq(lambda x: T(x) - 1 / q, x0 + 1e-12, 50) for q in qs])
    dT = lambda x: (T(x + 1e-6) - T(x - 1e-6)) / 2e-6
    I = np.array([quad(lambda x: S(x) * dT(x), x, 1)[0] for x in xq])   # int_{T_0/q}^{T_0} S dT  (T_0 = 1)
    tau = qs * I                                                       # (q-1) S_q / S_1
    Sq = np.where(np.abs(qs - 1) > 1e-9, tau / np.where(np.abs(qs - 1) > 1e-9, qs - 1, 1), 1.0)
    alpha = I + S(xq) / qs                                             # exact: d/dq (q I), using T(x_q) = 1/q
    return xq, Sq, tau, alpha, S(xq)


def part4():
    qs = np.linspace(0.3, 12, 600)
    out = {}
    lam_list = [-7 / 36, -0.1, 0.0, 0.05, 0.09]
    print("[4] Gauss-Bonnet d=4:  lambda_GB -> a = lambda f_inf = (1 - sqrt(1 - 4 lambda))/2")
    t2E = None
    for lam in lam_list:
        a = (1 - np.sqrt(1 - 4 * lam)) / 2
        xq, Sq, tau, alpha, Sx = gb_spectrum(a, qs=qs)
        i1 = np.argmin(np.abs(qs - 1))
        t2 = np.gradient(np.gradient(tau, qs), qs)[i1] / 2       # tau = t1 (q-1) + t2 (q-1)^2 + ...
        extra = ""
        if lam == 0.0:
            t2E = t2
            xe = (1 + np.sqrt(1 + 8 * qs ** 2)) / (4 * qs)
            Se = qs / (2 * (qs - 1)) * (2 - xe ** 2 * (1 + xe ** 2))
            m = np.abs(qs - 1) > 1e-3
            extra = f"; Einstein check max|S_q - HMSY closed form| = {np.max(np.abs(Sq - Se)[m]):.1e}"
        neg = qs[Sx < 0]
        out[lam] = (alpha, Sx, a)
        print(f"    lambda={lam:+.4f} (a={a:+.4f}): t2 = {t2:.5f}" + extra
              + (f"; S~_q < 0 for q > {neg.min():.2f}" if len(neg) else ""))
    table, cols, names = [], [qs], ["q"]
    for lam in lam_list:
        a = out[lam][2]
        xq, Sq, tau, alpha, Sx = gb_spectrum(a, qs=qs)
        t2 = np.gradient(np.gradient(tau, qs), qs)[np.argmin(np.abs(qs - 1))] / 2
        table.append([lam, a, t2, t2 / t2E, (1 - 2 * a) / (1 - 6 * a)])
        cols += [xq, tau, alpha, Sx]; names += [f"x_q_lam{lam:+.4f}", f"tau_lam{lam:+.4f}",
                                                 f"alpha_lam{lam:+.4f}", f"f_lam{lam:+.4f}"]
        if lam != 0.0:
            print(f"    t2/t2_Einstein = {t2 / t2E:.5f}   vs  (C_T/a)_GB/(C_T/a)_E = (1-2a)/(1-6a) = {(1 - 2 * a) / (1 - 6 * a):.5f}")
    T = np.array(table)
    savedata("gauss_bonnet_t2", ["lambda_GB", "a", "t2", "t2_over_t2_Einstein", "CT_over_a_ratio"],
             [T[:, i] for i in range(5)], ["d=4 Gauss-Bonnet hyperbolic black-hole cosmic branes (Table 1 of the paper)"])
    savedata("gauss_bonnet_branes", names, cols, ["d=4; tau, alpha, f in units of S_1; f = refined Renyi / S_1"])
    return out


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    qs, al_e, f_e, al_w, f_w, eps = part1()
    mk = part2()
    part3()
    gb = part4()

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    a0 = ax[0, 0]
    a0.plot(al_w, f_w, "k", label="wave function f_wave (PRBM, annealed)")
    a0.axhline(0, color="grey", lw=0.5)
    a0.set_xlabel("alpha"); a0.set_ylabel("f_wave(alpha)")
    a0.set_title("PRBM (b=1): wave-function spectrum vs entanglement spectrum (inset)")
    ins = a0.inset_axes([0.58, 0.12, 0.38, 0.38])
    qq = np.linspace(0.3, 8, 300)
    ins.plot(al_e, f_e, "tab:red", label="PRBM f_ent / S_1")
    ins.plot((1 + 1 / qq ** 2) / 2, 1 / qq, "k:", label="clean CFT (CL)")
    ins.set_xlim(0.4, 3); ins.set_ylim(0, 3); ins.legend(fontsize=6); ins.set_title("entanglement f_ent", fontsize=8)
    a0.legend(fontsize=8, loc="upper left")

    a1 = ax[0, 1]
    for l2, (a, f) in mk.items():
        a1.plot(a, f, label=f"lambda_2 = {l2}" + (" (reducible)" if l2 == 1.0 else ""))
    a1.set_xlabel("alpha"); a1.set_ylabel("f(alpha)"); a1.legend(fontsize=8)
    a1.set_title("Markov cascade (annealed over sectors, so f < 0 is possible):\n"
                 "analytic for lambda_2 < 1, linear segment only at lambda_2 = 1")

    a2 = ax[1, 0]
    q = np.linspace(-1, 2, 300); u = q * (1 - q)
    a2.plot(q, 2 * (0.1291 + 0.0029 * (q - 0.5) ** 2), "k", label="Delta_q / (q(1-q)), EMM 2008")
    a2.axhline(2 * 0.1291, ls="--", color="tab:orange", label="parabolic (one loop / WZW)")
    a2.set_xlabel("q"); a2.set_ylabel("Delta_q / q(1-q)"); a2.legend(fontsize=8)
    a2.set_title("Quantum Hall: c1 u + c2 u^2, u = q(1-q);  c2 = -2 b1 (quartic Casimir)")

    a3 = ax[1, 1]
    for lam, (alpha, Sx, a) in gb.items():
        a3.plot(alpha[5:-5], Sx[5:-5], label=f"lambda_GB = {lam:+.3f}")
    a3.axhline(0, color="grey", lw=0.5)
    a3.set_xlim(0, 6); a3.set_ylim(-0.3, 4.5); a3.set_xlabel("alpha (units of S_1)"); a3.set_ylabel("f(alpha) = S~_q / S_1")
    a3.set_title("Gauss-Bonnet cosmic branes, d = 4"); a3.legend(fontsize=8)

    savefig(fig, "frontier2_deepdive")
    print(f"figures written to {FIG_DIR}/, data to {DATA_DIR}/")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
