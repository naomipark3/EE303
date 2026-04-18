"""
EE303 Problem 2(b): Multilayer Reflection and Transmission Coefficients
========================================================================

A plane wave at 1 GHz, linearly polarized along x-hat and with amplitude
1 V/m, is normally incident on the stack:

    Air  ->  Dry Skin (2 mm)  ->  Infiltrated Fat (5 mm)  ->  Muscle

We compute the generalized (multilayer) reflection coefficient R_tilde and
transmission coefficient S at each of the three interfaces.

Convention: e^{+j*omega*t} (engineering), so lossy media have Im(eps_r) < 0
and waves traveling in +z have the form e^{-j*k*z}.
"""

import numpy as np

#Physical constants and operating point
eps_0 = 8.8541878128e-12    #F/m
mu_0  = 1.25663706212e-6    #H/m
c0    = 1.0 / np.sqrt(eps_0 * mu_0)
eta_0 = np.sqrt(mu_0 / eps_0)

f = 1.0e9                    #1 GHz
omega = 2.0 * np.pi * f
k0 = omega / c0

print("Operating point/initial conditions (i.e. variables defined in the problem statement):")
print(f"f = {f:.3e} Hz")
print(f"omega = {omega:.4e} rad/s")
print(f"k0 = {k0:.4f} rad/m")
print(f"eta_0 = {eta_0:.3f} Ohm")
print(f"omega*eps_0 = {omega*eps_0:.4e} S/m")

#4-term Cole-Cole model
def cole_cole(omega, params):
    """
    Evaluate the 4-term Cole-Cole model (engineering convention). We solved for these
    quantities in part (a), but we will solve them here for additional precision (and we
    can use these quantities to solve for the reflection and transmittion coefficient
    at each interface).

    eps_r = eps_inf + sum_n [ delta_eps_n / (1 + (j*omega*tau_n)^(1-alpha_n)) ]
            - j * sigma / (omega * eps_0)

    params: dict with keys
        eps_inf, sigma,
        deltas = [de1, de2, de3, de4],
        taus   = [t1, t2, t3, t4],        (seconds)
        alphas = [a1, a2, a3, a4].
    """
    eps_r = params['eps_inf'] + 0j
    for de, tau, a in zip(params['deltas'], params['taus'], params['alphas']):
        if de == 0.0:
            continue
        eps_r += de / (1.0 + (1j * omega * tau) ** (1.0 - a))
    eps_r -= 1j * params['sigma'] / (omega * eps_0)
    return eps_r


# Cole-Cole parameters from Gabriel et al. (lecture 3, slide 25).
# Times converted to seconds: tau1 in ps, tau2 in ns, tau3 in us, tau4 in ms.
skin = {
    'eps_inf': 4.0,
    'deltas':  [32.0, 1100.0, 0.0, 0.0],
    'taus':    [7.23e-12, 32.48e-9, 0.0, 0.0],
    'alphas':  [0.00, 0.20, 0.0, 0.0],
    'sigma':   0.0002,
}
fat = {
    'eps_inf': 2.5,
    'deltas':  [9.0, 35.0, 3.3e4, 1.0e7],
    'taus':    [7.96e-12, 15.92e-9, 159.15e-6, 15.915e-3],
    'alphas':  [0.20, 0.10, 0.05, 0.01],
    'sigma':   0.0350,
}
muscle = {
    'eps_inf': 4.0,
    'deltas':  [50.0, 7000.0, 1.2e6, 2.5e7],
    'taus':    [7.23e-12, 353.68e-9, 318.31e-6, 2.274e-3],
    'alphas':  [0.10, 0.10, 0.10, 0.00],
    'sigma':   0.200,
}

# Build the per-layer arrays.
# Index convention: 0 = air, 1 = skin, 2 = fat, 3 = muscle.
# Thickness is meaningful only for the finite (middle) layers.
eps_r = np.array([
    1.0 + 0j,
    cole_cole(omega, skin),
    cole_cole(omega, fat),
    cole_cole(omega, muscle),
])
thickness = np.array([np.inf, 2.0e-3, 5.0e-3, np.inf])   # meters
layer_names = ['Air', 'Dry Skin', 'Infiltrated Fat', 'Muscle']

print()
print("Relative permittivity at 1 GHz (from part a):")
for name, e in zip(layer_names, eps_r):
    print(f"{name:18s}  eps_r = {e.real:8.4f} + ({e.imag:+8.4f})j")

#Intrinsic impedance (eta) and wavenumber (k) in each layer.
#eta_n = eta_0 / sqrt(eps_r_n),  k_n = k0 * sqrt(eps_r_n).
#numpy's sqrt on a complex number returns the principal root, which is the
#physically correct branch (positive real part -> +z propagation that decays).
eta = eta_0 / np.sqrt(eps_r)
k   = k0 * np.sqrt(eps_r)

print()
print("Intrinsic impedance and wavenumber per layer:")
for name, et, kn in zip(layer_names, eta, k):
    print(f"  {name:18s}  eta = {et.real:8.3f} + ({et.imag:+8.3f})j Ohm"
          f"    k = {kn.real:9.3f} + ({kn.imag:+9.3f})j rad/m")

# Calculate R and 
# Interface n is between layer n and layer n+1, n = 0, 1, 2.
# R_{n,n+1} = (eta_{n+1} - eta_n) / (eta_{n+1} + eta_n)
# T_{n,n+1} = 2*eta_{n+1} / (eta_{n+1} + eta_n)
N_interfaces = 3
R = np.zeros(N_interfaces, dtype=complex)
T = np.zeros(N_interfaces, dtype=complex)

for n in range(N_interfaces):
    R[n] = (eta[n+1] - eta[n]) / (eta[n+1] + eta[n])
    T[n] = 2.0 * eta[n+1]       / (eta[n+1] + eta[n])

interface_names = [
    'Air  -> Skin   (0,1)',
    'Skin -> Fat    (1,2)',
    'Fat  -> Muscle (2,3)',
]

print()
print("Single-interface coefficients (bare R, T)")
for name, R_val, T_val in zip(interface_names, R, T):
    print(f"  {name}:  R = {R_val.real:+.6f} + ({R_val.imag:+.6f})j"
          f"   T = {T_val.real:+.6f} + ({T_val.imag:+.6f})j")

# Sanity check: T = 1 + R must hold at every interface.
print()
print("  Sanity check (should be ~0):  max |T - (1+R)| =",
      np.max(np.abs(T - (1.0 + R))))

# Generalized multilayer coefficients, recursion from the bottom up.
#
# Bottom interface (fat->muscle): muscle is semi-infinite, so nothing reflects
# back from below.  R_tilde = R,  S = T.
#
# Moving up one interface, for n = N_interfaces-2, ..., 0:
#   phase = exp(-j * 2 * k_{n+1} * thickness_{n+1})
#   R_tilde_{n,n+1} = (R_{n,n+1} + R_tilde_{n+1,n+2} * phase)
#                     / (1 + R_{n,n+1} * R_tilde_{n+1,n+2} * phase)
#   S_{n,n+1} = T_{n,n+1} / (1 + R_{n,n+1} * R_tilde_{n+1,n+2} * phase)
R_tilde = np.zeros(N_interfaces, dtype=complex)
S       = np.zeros(N_interfaces, dtype=complex)

# Base case: bottom-most interface.
R_tilde[-1] = R[-1]
S[-1]       = T[-1]

# Recursion upward.
for n in range(N_interfaces - 2, -1, -1):
    # Round-trip propagation factor through layer n+1.
    phase = np.exp(-1j * 2.0 * k[n+1] * thickness[n+1])
    denom = 1.0 + R[n] * R_tilde[n+1] * phase
    R_tilde[n] = (R[n] + R_tilde[n+1] * phase) / denom
    S[n]       = T[n] / denom

print()
print("Reflection coefficient R_tilde_{n,n+1} and transmission coefficient S_{n,n+1}")
for name, Rt, Sn in zip(interface_names, R_tilde, S):
    print(f"{name}:")
    print(f"R_tilde = {Rt.real:+.6f} + ({Rt.imag:+.6f})j"
          f"|R_tilde| = {abs(Rt):.4f}")
    print(f"S = {Sn.real:+.6f} + ({Sn.imag:+.6f})j"
          f"|S| = {abs(Sn):.4f}")

# Power sanity check at the top interface (air -> stack).
# Incident medium is lossless (air), so:
#   Fraction of incident power reflected = |R_tilde_{0,1}|^2.
#   Fraction not reflected (= transmitted into stack + dissipated) = 1 - |R_t|^2.
# Inside the stack, power is genuinely dissipated in the lossy layers, so we
# cannot cleanly split "transmitted to muscle" from "absorbed in skin+fat"
# without integrating Poynting's theorem. The top-level |R|^2 budget is the
# most useful single sanity check.
# print()
# print("Power budget at the air-skin interface:")
# P_refl = abs(R_tilde[0]) ** 2
# print(f"|R_tilde_01|^2  = {P_refl:.4f}   (fraction of incident power reflected)")
# print(f"1 - |R_tilde_01|^2 = {1 - P_refl:.4f}   (enters + is dissipated in the stack)")