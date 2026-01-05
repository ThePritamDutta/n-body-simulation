import numpy as np

# Gravitational constant
G_DEFAULT = 6.67430e-11

def get_acc(state, masses, G=G_DEFAULT, epsilon=1e4):
    """
    Computes accelerations using vectorized broadcasting.
    N^2 memory usage, but extremely fast in Python for N < 500.
    """
    N = len(masses)
    # state is (6N,), we only need positions (x,y,z)
    positions = state.reshape((N, 6))[:, :3]

    # Matrix of displacement vectors: r_ij = r_j - r_i
    # Shape: (N, N, 3)
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # Squared distance with Plummer softening to avoid 1/0
    dist_sq = np.sum(r_ij**2, axis=-1) + epsilon**2

    # Ignore self-interaction (diagonal)
    np.fill_diagonal(dist_sq, np.inf)

    # a_i = G * sum( m_j * r_ij / dist^3 )
    inv_dist3 = dist_sq ** (-1.5)
    
    # We broadcast masses to match the (N, N, 3) force calculation
    acc = G * np.sum(
        r_ij * inv_dist3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis],
        axis=1
    )

    return acc.reshape(-1)

def verlet_step(t0, y0, masses, tf, dt, G=G_DEFAULT, epsilon=1e4):
    """
    Standard Velocity Verlet (Symplectic). 
    This is generally better than RK4 for long-term orbital stability.
    """
    t = t0
    y = y0.copy()
    N = len(masses)

    t_list = [t]
    y_list = [y.copy()]

    # Initial acceleration for the first half-kick
    acc = get_acc(y, masses, G, epsilon).reshape((N, 3))

    while t < tf:
        dt_step = min(dt, tf - t)

        y_mat = y.reshape((N, 6))
        v = y_mat[:, 3:6]

        # 1. Half Kick: v(t + dt/2) = v(t) + a(t) * dt/2
        v += 0.5 * acc * dt_step

        # 2. Drift: r(t + dt) = r(t) + v(t + dt/2) * dt
        y_mat[:, :3] += v * dt_step

        # 3. Get new acceleration at the updated position
        acc = get_acc(y, masses, G, epsilon).reshape((N, 3))

        # 4. Kick: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        v += 0.5 * acc * dt_step

        t += dt_step
        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)

def rk4(f, t0, y0, tf, dt):
    """
    Classic 4th order Runge-Kutta. 
    High local accuracy, but expect energy drift over many orbits.
    """
    t = t0
    y = y0.copy()

    t_list = [t]
    y_list = [y.copy()]

    while t < tf:
        dt_step = min(dt, tf - t)

        k1 = dt_step * f(t, y)
        k2 = dt_step * f(t + 0.5 * dt_step, y + 0.5 * k1)
        k3 = dt_step * f(t + 0.5 * dt_step, y + 0.5 * k2)
        k4 = dt_step * f(t + dt_step, y + k3)

        y += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt_step

        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)

def rk45(f, t0, y0, tf, dt, tol):
    """
    Adaptive Step-size RK45 (Fehlberg).
    Useful if bodies get very close and you need higher resolution temporarily.
    """
    dt_min = 1e-6
    dt_max = dt
    current_dt = dt # Track the moving step size separately

    t = t0
    y = y0.copy()

    t_list = [t]
    y_list = [y.copy()]

    while t < tf:
        dt_step = min(current_dt, tf - t)

        # Butcher Tableau coefficients for RKF45
        k1 = dt_step * f(t, y)
        k2 = dt_step * f(t + dt_step/4, y + k1/4)
        k3 = dt_step * f(t + 3*dt_step/8, y + 3*k1/32 + 9*k2/32)
        k4 = dt_step * f(t + 12*dt_step/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = dt_step * f(t + dt_step, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = dt_step * f(t + dt_step/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

        # Compute 4th and 5th order solutions
        y4 = y + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - k5/5
        y5 = y + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6

        # Estimate local truncation error
        error = np.sqrt(np.mean(((y5 - y4) / np.maximum(np.abs(y), 1.0))**2))
        error = max(error, 1e-16)

        # Adaptive step size logic
        if error <= tol or dt_step <= dt_min:
            # Step accepted
            t += dt_step
            y = y5
            t_list.append(t)
            y_list.append(y.copy())
            
        # Update dt for next iteration based on error
        current_dt *= 0.9 * (tol / error)**0.2
        current_dt = np.clip(current_dt, dt_min, dt_max)

    return np.array(t_list), np.array(y_list)