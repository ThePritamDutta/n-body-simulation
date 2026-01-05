import numpy as np

# Standard Gravitational Constant
G = 6.67430e-11

def total_energy_drift(y_history, masses, N, eps=1e4):
    """
    Tracks the stability of the system by calculating 
    (E(t) - E(0)) / |E(0)| over time.
    """
    
    energies = []

    for state in y_history:
        # Pull out positions and velocities
        reshaped_state = state.reshape((N, 6))
        pos = reshaped_state[:, 0:3]
        vel = reshaped_state[:, 3:6]

        # Shift to Center of Mass frame to avoid 'fake' drifts
        total_m = np.sum(masses)
        r_cm = np.sum(masses[:, None] * pos, axis=0) / total_m
        v_cm = np.sum(masses[:, None] * vel, axis=0) / total_m

        r = pos - r_cm
        v = vel - v_cm

        # Kinetic Energy: Sum(1/2 * m * v^2)
        v_sq = np.sum(v**2, axis=1)
        K = 0.5 * np.sum(masses * v_sq)

        # Potential Energy: Sum(-G * m1 * m2 / sqrt(r^2 + eps^2))
        # Optimized approach: calculate all pairs at once
        U = 0.0
        
        # We calculate displacement vectors for all pairs
        # r[:, None, :] - r[None, :, :] creates an (N, N, 3) matrix of r_ij
        dx = r[:, None, 0] - r[None, :, 0]
        dy = r[:, None, 1] - r[None, :, 1]
        dz = r[:, None, 2] - r[None, :, 2]
        
        # Compute softened distances matrix (N x N)
        dist_matrix = np.sqrt(dx**2 + dy**2 + dz**2 + eps**2)
        
        # Calculate product of masses (N x N)
        m_i_m_j = masses[:, None] * masses[None, :]
        
        # Energy matrix
        energy_matrix = -G * m_i_m_j / dist_matrix
        
        # Sum only the upper triangle (to avoid double counting and self-interaction)
        U = np.sum(np.triu(energy_matrix, k=1))

        energies.append(K + U)

    E = np.array(energies)
    
    # Calculate relative drift from initial energy
    E0 = E[0]
    # Check for E0 = 0 to avoid division by zero (unlikely in N-body)
    if abs(E0) < 1e-20:
        return np.zeros_like(E)
        
    dE_rel = (E - E0) / abs(E0)
    return dE_rel


def Angular_momentum(y, masses, N):
    """
    Computes total angular momentum magnitude (L) and relative drift.
    Ensures calculations are done relative to the system's center of mass.
    """

    states = y.reshape((-1, N, 6))
    L_vec_list = []

    for state in states:
        r_raw = state[:, :3]
        v_raw = state[:, 3:6]

        # Again, shift to COM frame
        total_m = np.sum(masses)
        r_cm = np.sum(masses[:, None] * r_raw, axis=0) / total_m
        v_cm = np.sum(masses[:, None] * v_raw, axis=0) / total_m

        r = r_raw - r_cm
        v = v_raw - v_cm

        # L = r x p = r x (m*v)
        # np.cross handles the cross product for each body's row
        L_individual = np.cross(r, masses[:, None] * v)
        L_total = np.sum(L_individual, axis=0)
        L_vec_list.append(L_total)

    L_vecs = np.array(L_vec_list) 
    L_mag = np.linalg.norm(L_vecs, axis=1)

    # Calculate relative drift from initial L
    L0 = L_mag[0]
    if abs(L0) < 1e-20:
        return L_mag, np.zeros_like(L_mag)
        
    delta_L_over_L = (L_mag - L0) / abs(L0)

    return L_mag, delta_L_over_L