# N-body Simulation

This repository contains a first-principles implementation of **gravitational N-body simulations** in Python. It is designed for computational astrophysics projects and numerical experimentation with few- to many-body gravitational systems.

## Features

- **Multiple numerical integrators**:
  - Euler
  - Leapfrog
  - Runge-Kutta 4 (RK4)
  - Runge-Kutta-Fehlberg 45 (RK45)
- **System validation**:
  - Two-body and three-body problem tests
  - Conservation of total energy and angular momentum
- **Visualization**:
  - 2D and 3D plots of trajectories
  - Optional animation output (GIF / MP4)
- **Extensible framework**:
  - Can scale to larger N using additional optimization techniques (Barnes-Hut tree, etc.)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ThePritamDutta/n-body-simulation.git
   cd n-body-simulation
