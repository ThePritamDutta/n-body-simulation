import numpy as np

# Gravitational constant
G = 6.67430e-11

def bodies(N, seed=None):
    """
    Generate virialized initial conditions for an N-body gravitational system.
    """

    if seed is not None:
        np.random.seed(seed)

    # ---------------------------------------------------------
    # 1. Mass distribution
    # ---------------------------------------------------------
    masses = np.random.uniform(1e20, 1e25, size=N)

    # ---------------------------------------------------------
    # 2. Position distribution
    # ---------------------------------------------------------
    pos_scale = 3e8  # meters
    positions = np.random.uniform(-pos_scale, pos_scale, size=(N, 3))

    # ---------------------------------------------------------
    # 3. Initial random velocities (unscaled)
    # ---------------------------------------------------------
    vel_scale = 1.0  # temporary scale, will be rescaled
    velocities = np.random.uniform(-vel_scale, vel_scale, size=(N, 3))

    # ---------------------------------------------------------
    # 4. Compute potential energy U
    # ---------------------------------------------------------
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            U -= G * masses[i] * masses[j] / r

    # ---------------------------------------------------------
    # 5. Compute kinetic energy K
    # ---------------------------------------------------------
    K = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    # ---------------------------------------------------------
    # 6. Virialize velocities: enforce 2K + U = 0
    # ---------------------------------------------------------
    scale = np.sqrt(abs(U) / (2 * K))
    velocities *= scale

    # ---------------------------------------------------------
    # 7. Remove center-of-mass motion
    # ---------------------------------------------------------
    total_mass = np.sum(masses)
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    velocities -= v_cm

    # ---------------------------------------------------------
    # 8. Assemble state vector
    # ---------------------------------------------------------
    state = np.zeros((N, 6))
    state[:, :3] = positions
    state[:, 3:6] = velocities

    return state.flatten(), masses
