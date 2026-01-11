
import numpy as np


def total_energy(y_history, masses, N, eps=1e4):
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
                dist = np.sqrt(dx*dx + dy*dy + dz*dz + eps*eps)
                U -= G * masses[i] * masses[j] / dist

        energies[t] = K + U

    E0 = energies[0]
    dE_rel = (energies - E0) / np.abs(E0)

    return dE_rel

def Angular_momentum(y, masses, N):
    states = y.reshape((-1, N, 6))
    T = states.shape[0]

    L_mag = np.zeros(T)

    for t in range(T):
        state = states[t]
        Lx = 0.0
        Ly = 0.0
        Lz = 0.0

        for i in range(N):
            rx, ry, rz = state[i, 0], state[i, 1], state[i, 2]
            vx, vy, vz = state[i, 3], state[i, 4], state[i, 5]

            mx = masses[i] * vx
            my = masses[i] * vy
            mz = masses[i] * vz

            # r × (m v)
            Lx += ry * mz - rz * my
            Ly += rz * mx - rx * mz
            Lz += rx * my - ry * mx

        L_mag[t] = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)

    L0 = L_mag[0]
    delta_L_over_L = np.abs(L_mag - L0) / np.abs(L0)

    return L_mag, delta_L_over_L

