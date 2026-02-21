# 🌌 N-Body Gravitational Simulation

## Velocity Verlet vs RK4 vs RK45 Integrators

**Authors:** Pritam Dutta & Akshat Chakarverty\
**Institute:** Indian Association for the Cultivation of Science\
**Year:** 2025

------------------------------------------------------------------------

## 📌 Overview

This project presents a direct Newtonian **N-body gravitational
simulator** developed to systematically compare:

-   🔵 Velocity Verlet (Symplectic)
-   🟢 Runge--Kutta 4 (RK4)
-   🟣 Runge--Kutta 45 (RK45)

The study focuses on:

-   Energy conservation\
-   Angular momentum conservation\
-   Runtime scaling\
-   Long-term numerical stability\
-   Solar system evolution

Simulations range from **2-body systems** to **500-body systems**,
including a **248-year Solar System simulation**.

------------------------------------------------------------------------

## 🧠 Physics Background

We numerically solve Newton's equations of motion:

m_i d²r_i/dt² = Σ G m_i m_j (r_j − r_i) / (\|r_j − r_i\|² + ε²)\^(3/2)

Key features:

✔ Softening parameter (ε)\
✔ Virialized initial conditions (2K + U = 0)\
✔ Center-of-mass frame correction\
✔ Elastic collision handling\
✔ Energy & angular momentum diagnostics\
✔ Lyapunov exponent estimation

------------------------------------------------------------------------

## 🚀 Features

-   Direct O(N²) force evaluation\
-   Symplectic and non-symplectic integrators\
-   Energy & angular momentum drift analysis\
-   Runtime scaling study\
-   Solar system simulation\
-   GIF/animation support\
-   HDF5 output support

------------------------------------------------------------------------

## 📊 Key Results

### Energy Conservation

Velocity Verlet shows bounded oscillatory error (symplectic behavior),\
while RK4 and RK45 show monotonic drift over long simulations.

### Runtime Scaling

Runtime follows:

T(N) ∝ N²

Force evaluation dominates computational cost.

------------------------------------------------------------------------

## 🪐 Solar System Simulation

-   Simulated for 248 years\
-   Time step: 8000 seconds\
-   Sun fixed at origin (future work: barycentric correction)

------------------------------------------------------------------------

## 🛠 Installation

``` bash
git clone https://github.com/Akshat-Chakarverty/n-body-simulation
cd n-body-simulation
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶ Running Example

``` python
python main.py --N 100 --integrator verlet --dt 1000 --T 100
```

------------------------------------------------------------------------

## 📦 Dependencies

-   numpy\
-   matplotlib\
-   h5py\
-   pillow\
-   ffmpeg

------------------------------------------------------------------------

## 📈 Energy Drift Formula

Relative energy drift:

ΔE = \|E(T) − E(0)\| / \|E(0)\|

Drift rate:

γ = (1/T) \|E(T) − E(0)\| / \|E(0)\|

------------------------------------------------------------------------

## 🔮 Future Work

-   Barnes--Hut Tree Code (O(N log N))\
-   MPI/OpenMP parallelization\
-   Barycentric Solar System model\
-   Adaptive symplectic integrators\
-   Cosmological simulations

------------------------------------------------------------------------

## 📜 Citation

Dutta, P. & Chakarverty, A. (2025)\
N-Body Gravitational Simulations Using RK and Verlet Integrators\
Indian Association for the Cultivation of Science

------------------------------------------------------------------------

## 🌟 Final Note

Higher local accuracy does not guarantee long-term physical fidelity.\
Symplectic structure preservation is essential for gravitational
systems.
