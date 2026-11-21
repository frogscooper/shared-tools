# stellar-density

```bash
stellar-density 4.411937 --a-rs 9.8785 --mstar 1.28 --rstar 1.21     # WASP-62 b: rho 0.6645 rho_sun, ratio 0.920
stellar-density 365.25636 --a-au 1 --rstar 1                         # sanity check: 1.000 rho_sun
```

`rho = 3π/(G P²) (a/Rs)³`, all in cgs (G = 6.674e-8, P in s). If you start from a in AU, then
`a/Rs = a_AU × 215.032 / R*`. That is the (AU/Rsun)³ ≈ 9.94e6 factor, and forgetting it is the usual bug.
ρ☉ = 1.4098 g/cm³. With `--mstar` and `--rstar` it also reports the TIC density M/R³ and the ratio.
