import numpy as np



def bodies(N, G=6.67430e-11):
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    masses = np.zeros(N)
    radii = np.zeros(N)

    # -----------------------------------------
    # 1. Random positions, velocities, masses
    # -----------------------------------------
    for i in range(N):
        positions[i, 0] = np.random.uniform(-1.6e9, 1.6e9)
        positions[i, 1] = np.random.uniform(-1.6e9, 1.6e9)
        positions[i, 2] = np.random.uniform(-1.6e9, 1.6e9)

        velocities[i, 0] = np.random.uniform(-500, 500)
        velocities[i, 1] = np.random.uniform(-500, 500)
        velocities[i, 2] = np.random.uniform(-500, 500)

        masses[i] = np.random.uniform(1e20, 1e25)

        radii[i] = np.random.uniform(1e6,5e6)

    # -----------------------------------------
    # 2. Potential energy U (double loop)
    # -----------------------------------------
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]

            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            U -= G * masses[i] * masses[j] / r

    # -----------------------------------------
    # 3. Kinetic energy K (loop)
    # -----------------------------------------
    K = 0.0
    for i in range(N):
        v2 = (
            velocities[i, 0] * velocities[i, 0] +
            velocities[i, 1] * velocities[i, 1] +
            velocities[i, 2] * velocities[i, 2]
        )
        K += 0.5 * masses[i] * v2

    # -----------------------------------------
    # 4. Virialize velocities (2K + U = 0)
    # -----------------------------------------
    scale = np.sqrt(-U / (2.0 * K))
    for i in range(N):
        velocities[i, 0] *= scale
        velocities[i, 1] *= scale
        velocities[i, 2] *= scale

    # -----------------------------------------
    # 5. Remove center-of-mass motion
    # -----------------------------------------
    total_mass = 0.0
    vcm_x = 0.0
    vcm_y = 0.0
    vcm_z = 0.0

    for i in range(N):
        total_mass += masses[i]
        vcm_x += masses[i] * velocities[i, 0]
        vcm_y += masses[i] * velocities[i, 1]
        vcm_z += masses[i] * velocities[i, 2]

    vcm_x /= total_mass
    vcm_y /= total_mass
    vcm_z /= total_mass

    for i in range(N):
        velocities[i, 0] -= vcm_x
        velocities[i, 1] -= vcm_y
        velocities[i, 2] -= vcm_z

    # -----------------------------------------
    # 6. Assemble state vector
    # -----------------------------------------
    state = np.zeros((N, 6))
    for i in range(N):
        state[i, 0] = positions[i, 0]
        state[i, 1] = positions[i, 1]
        state[i, 2] = positions[i, 2]
        state[i, 3] = velocities[i, 0]
        state[i, 4] = velocities[i, 1]
        state[i, 5] = velocities[i, 2]

    return state.ravel(), masses,radii


