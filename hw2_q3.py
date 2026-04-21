"""
Same geometry as problem 2 (air -> skin 2mm -> fat 5mm -> muscle), but
the plane wave is now acoustic at 1 MHz.  Compute the reflection coefficient r_tilde
and transmission coefficients at each of the three interfaces, using acoustic impedance
Z = rho*c in place of EM intrinsic impedance eta, and acoustic wavenumber k = w/c.
The recursion implementation is identical to problem 2(b):
r_tilde_{n,n+1} = (r_{n,n+1} + r_tilde_{n+1,n+2} * phase) / (1 + r_{n,n+1} * r_tilde_{n+1,n+2} * phase)
s_{n,n+1} = t_{n,n+1} / (1 + r_{n,n+1} * r_tilde_{n+1,n+2} * phase) with base case r_tilde at the bottom = r.
*NOTE: the lecture 3 slide 72 formulas R = ((Z1-Z2)/(Z1+Z2))^2 and T = 4 Z1 Z2 / (Z1+Z2)^2
are POWER coefficients. For the recursion we need the PRESSURE
amplitude coefficients r = (Z2-Z1)/(Z2+Z1) and t = 2 Z2 / (Z2+Z1). This is the same
structure as EM, with Z playing the role of eta.
"""

import numpy as np

f = 1.0e6                    #1 MHz = 1*10^6 Hz
omega = 2.0 * np.pi * f #\omega = 2pif = 2pi*(1*10^6) Hz

print("Operating point")
print(f"f = {f:.3e} Hz")
print(f"omega = {omega:.4e} rad/s")

#Acoustic properties from the problem statement.
#Index convention: 0 = air, 1 = skin, 2 = fat, 3 = muscle.
#Z (kg m^-2 s^-1), rho (kg m^-3), c (m/s).
Z  = np.array([0.0004e6, 1.6e6, 1.35e6, 1.7e6])      #acoustic impedance, as given in the table
rho = np.array([1.2, 1000.0, 920.0, 1070.0])      #density
c = np.array([330.0, 1600.0, 1450.0, 1580.0])    #speed
thickness = np.array([np.inf, 2.0e-3, 5.0e-3, np.inf])
layer_names = ['Air', 'Skin', 'Fat', 'Muscle']

#Wavenumber in each layer. Fully real since medium is lossless.
k = omega / c

print()
print("Acoustic properties for each layer")
for name, Zn, cn, kn in zip(layer_names, Z, c, k):
    print(f"{name:8s}  Z = {Zn:.4e} Rayl   c = {cn:6.1f} m/s   k = {kn:8.2f} rad/m")

#Per-interface pressure reflection and transmission coefficients
#r_{n,n+1} = (Z_{n+1} - Z_n) / (Z_{n+1} + Z_n)
#t_{n,n+1} = 2 * Z_{n+1}    / (Z_{n+1} + Z_n)
#These satisfy t = 1 + r (pressure continuity across the boundary).
N_interfaces = 3
r = np.zeros(N_interfaces, dtype=complex)
t = np.zeros(N_interfaces, dtype=complex)

for n in range(N_interfaces):
    r[n] = (Z[n+1] - Z[n]) / (Z[n+1] + Z[n])
    t[n] = 2.0 * Z[n+1] / (Z[n+1] + Z[n])

interface_names = [
    'Air -> Skin (0,1 interface)',
    'Skin -> Fat (1,2 interface)',
    'Fat -> Muscle (2,3 interface)',
]

print()
print("Single-interface coefficients R, T")
for name, r1, t1 in zip(interface_names, r, t):
    print(f"{name}: R = {r1.real:+.6f} T = {t1.real:+.6f}")

print()
print("Sanity check (should be ~0):  max |t - (1+r)| =", np.max(np.abs(t - (1.0 + r))))

#R_tilde and S, recursion from the bottom up.
#Base case: bottom interface has muscle as semi-infinite, so r_tilde = r.
r_tilde = np.zeros(N_interfaces, dtype=complex)
s = np.zeros(N_interfaces, dtype=complex)

r_tilde[-1] = r[-1]
s[-1] = t[-1]

for n in range(N_interfaces - 2, -1, -1):
    phase = np.exp(-1j * 2.0 * k[n+1] * thickness[n+1])
    denom = 1.0 + r[n] * r_tilde[n+1] * phase
    r_tilde[n] = (r[n] + r_tilde[n+1] * phase) / denom
    s[n] = t[n] / denom

print()
print("R_tilde and S")
for name, rt, sn in zip(interface_names, r_tilde, s):
    print(f"{name}:")
    print(f"r_tilde = {abs(rt):.4f}")
    print(f"s = {abs(sn):.4f}")