"""
EE303 Problem 3: Acoustic Wave Propagation Through Air-Skin-Fat-Muscle
=======================================================================

Same geometry as problem 2 (air -> skin 2mm -> fat 5mm -> muscle), but
the plane wave is now acoustic at 1 MHz.  Compute the generalized
(multilayer) reflection coefficient r_tilde and transmission coefficient
s at each of the three interfaces, using acoustic impedance Z = rho*c
in place of EM intrinsic impedance eta, and acoustic wavenumber k = w/c.

The multilayer recursion is identical in form to problem 2(b):

    r_tilde_{n,n+1} = (r_{n,n+1} + r_tilde_{n+1,n+2} * phase)
                      / (1 + r_{n,n+1} * r_tilde_{n+1,n+2} * phase)
    s_{n,n+1}       = t_{n,n+1} / (1 + r_{n,n+1} * r_tilde_{n+1,n+2} * phase)

with base case r_tilde at the bottom = bare r, and phase = exp(-j*2*k*l)
for the layer directly below interface n.

Note: the slide-72 formulas R = ((Z1-Z2)/(Z1+Z2))^2 and T = 4 Z1 Z2 / (Z1+Z2)^2
are POWER coefficients.  For the multilayer recursion we need the PRESSURE
amplitude coefficients r = (Z2-Z1)/(Z2+Z1) and t = 2 Z2 / (Z2+Z1) -- same
structure as EM, with Z playing the role of eta.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Operating point
# -----------------------------------------------------------------------------
f = 1.0e6                    # 1 MHz
omega = 2.0 * np.pi * f #\omega = 2pif

print("=" * 72)
print("Operating point")
print("=" * 72)
print(f"f       = {f:.3e} Hz")
print(f"omega   = {omega:.4e} rad/s")

# Acoustic properties from the problem table.
# Index convention: 0 = air, 1 = skin, 2 = fat, 3 = muscle.
# Z (kg m^-2 s^-1), rho (kg m^-3), c (m/s).
Z   = np.array([0.0004e6, 1.6e6, 1.35e6, 1.7e6])      #acoustic impedance, as given in the table
rho = np.array([1.2,      1000.0, 920.0, 1070.0])      #density
c   = np.array([330.0,    1600.0, 1450.0, 1580.0])    #speed
thickness = np.array([np.inf, 2.0e-3, 5.0e-3, np.inf])
layer_names = ['Air', 'Skin', 'Fat', 'Muscle']

# Wavenumber in each layer.  Purely real since medium is lossless.
k = omega / c

# Self-consistency check: the table's Z should equal rho*c.
print()
print("=" * 72)
print("Table self-consistency:  does Z equal rho * c?")
print("=" * 72)
for name, Zn, rn, cn in zip(layer_names, Z, rho, c):
    print(f"  {name:8s}  Z_table = {Zn:.4e}   rho*c = {rn*cn:.4e}"
          f"   diff = {abs(Zn - rn*cn)/Zn*100:.2f}%")

print()
print("=" * 72)
print("Per-layer acoustic properties")
print("=" * 72)
for name, Zn, cn, kn in zip(layer_names, Z, c, k):
    print(f"  {name:8s}  Z = {Zn:.4e} Rayl   c = {cn:6.1f} m/s   k = {kn:8.2f} rad/m")

# -----------------------------------------------------------------------------
# Per-interface (bare) pressure reflection and transmission coefficients.
#   r_{n,n+1} = (Z_{n+1} - Z_n) / (Z_{n+1} + Z_n)
#   t_{n,n+1} = 2 * Z_{n+1}    / (Z_{n+1} + Z_n)
# These satisfy t = 1 + r (pressure continuity across the boundary).
# -----------------------------------------------------------------------------
N_interfaces = 3
r_fresnel = np.zeros(N_interfaces, dtype=complex)
t_fresnel = np.zeros(N_interfaces, dtype=complex)

for n in range(N_interfaces):
    r_fresnel[n] = (Z[n+1] - Z[n]) / (Z[n+1] + Z[n])
    t_fresnel[n] = 2.0 * Z[n+1]    / (Z[n+1] + Z[n])

interface_names = [
    'Air  -> Skin   (0,1)',
    'Skin -> Fat    (1,2)',
    'Fat  -> Muscle (2,3)',
]

print()
print("=" * 72)
print("Single-interface (bare) pressure coefficients r, t")
print("=" * 72)
for name, r, t in zip(interface_names, r_fresnel, t_fresnel):
    print(f"  {name}:  r = {r.real:+.6f}   t = {t.real:+.6f}")

print()
print("  Sanity check (should be ~0):  max |t - (1+r)| =",
      np.max(np.abs(t_fresnel - (1.0 + r_fresnel))))

# Also report the corresponding single-interface POWER coefficients for
# comparison with slide 72's formulas.  R_pow = r^2, T_pow = 1 - R_pow
# (lossless interface).  Equivalently T_pow = 4 Z1 Z2 / (Z1 + Z2)^2.
print()
print("  Single-interface POWER coefficients (for comparison with slide 72):")
for name, r, n in zip(interface_names, r_fresnel, range(N_interfaces)):
    R_pow = abs(r) ** 2
    T_pow = 4.0 * Z[n] * Z[n+1] / (Z[n] + Z[n+1]) ** 2
    print(f"    {name}:  R_pow = {R_pow:.6f}   T_pow = {T_pow:.6f}"
          f"   (sum = {R_pow + T_pow:.6f})")

# -----------------------------------------------------------------------------
# Generalized multilayer coefficients, recursion from the bottom up.
# Base case: bottom interface has muscle as semi-infinite, so r_tilde = r.
# -----------------------------------------------------------------------------
r_tilde = np.zeros(N_interfaces, dtype=complex)
s       = np.zeros(N_interfaces, dtype=complex)

r_tilde[-1] = r_fresnel[-1]
s[-1]       = t_fresnel[-1]

for n in range(N_interfaces - 2, -1, -1):
    phase = np.exp(-1j * 2.0 * k[n+1] * thickness[n+1])
    denom = 1.0 + r_fresnel[n] * r_tilde[n+1] * phase
    r_tilde[n] = (r_fresnel[n] + r_tilde[n+1] * phase) / denom
    s[n]       = t_fresnel[n] / denom

print()
print("=" * 72)
print("Round-trip phase through each finite layer  (2*k*l)")
print("=" * 72)
for idx, name in zip([1, 2], ['Skin', 'Fat']):
    rt_phase = 2.0 * k[idx] * thickness[idx]
    print(f"  {name:8s}  2*k*l = {rt_phase:.3f} rad  =  {rt_phase/np.pi:.3f} * pi")

print()
print("=" * 72)
print("Generalized (multilayer) pressure coefficients r_tilde, s")
print("=" * 72)
for name, rt, sn in zip(interface_names, r_tilde, s):
    print(f"  {name}:")
    print(f"       r_tilde = {rt.real:+.6f} + ({rt.imag:+.6f})j"
          f"     |r_tilde| = {abs(rt):.4f}")
    print(f"       s       = {sn.real:+.6f} + ({sn.imag:+.6f})j"
          f"     |s|       = {abs(sn):.4f}")

# -----------------------------------------------------------------------------
# Power budget at the top interface.
# -----------------------------------------------------------------------------
P_refl = abs(r_tilde[0]) ** 2
print()
print("=" * 72)
print("Power budget at the air-skin interface")
print("=" * 72)
print(f"  |r_tilde_01|^2    = {P_refl:.6f}   (fraction of incident power reflected)")
print(f"  1 - |r_tilde_01|^2 = {1 - P_refl:.6f}   (enters stack; lossless, so reaches muscle)")