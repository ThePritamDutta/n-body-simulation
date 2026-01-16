import numpy as np

def total_energy(y_history, masses, N):
    G = 6.67430e-11
    T = y_history.shape[0]

    energies = np.zeros(T)

    for t in range(T):
        state = y_history[t]
        reshaped_state = state.reshape((N, 6))

        positions = reshaped_state[:, 0:3]
        velocities = reshaped_state[:, 3:6]

        # Kinetic Energy
        v_sq = np.sum(velocities * velocities, axis=1)
        K = 0.5 * np.sum(masses * v_sq)

        # Potential Energy
        U = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                if dist < 1e-12:   # safety guard
                    continue

                U -= G * masses[i] * masses[j] / dist

        energies[t] = K + U

    return energies


def Angular_momentum(y_history, masses, N):
    T = y_history.shape[0]
    L_mag = np.zeros(T)

    for t in range(T):
        state = y_history[t].reshape((N, 6))
        Lx = 0.0
        Ly = 0.0
        Lz = 0.0

        for i in range(N):
            rx, ry, rz = state[i, 0], state[i, 1], state[i, 2]
            vx, vy, vz = state[i, 3], state[i, 4], state[i, 5]

            px = masses[i] * vx
            py = masses[i] * vy
            pz = masses[i] * vz

            Lx += ry * pz - rz * py
            Ly += rz * px - rx * pz
            Lz += rx * py - ry * px

        L_mag[t] = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)

    L0 = L_mag[0]
    if np.abs(L0) < 1e-20:
        delta_L_over_L = np.zeros_like(L_mag)
    else:
        delta_L_over_L = np.abs(L_mag - L0) / np.abs(L0)

    return L_mag, delta_L_over_L
