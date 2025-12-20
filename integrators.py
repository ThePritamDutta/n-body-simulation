import numpy as np


def get_acc(state, masses, G=6.67430e-11):

    N = len(masses)
    positions = state.reshape((N, 6))[:, :3]  # to get x,y,z
    acc = np.zeros((N, 3))

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec) + 1e4  # Softening from main.py

            force_mag = (G * masses[i] * masses[j]) / (dist**3)
            force_vec = r_vec * force_mag

            acc[i] += force_vec / masses[i]
            acc[j] -= force_vec / masses[j]

    return acc.flatten()


def verlet_step(t, y, masses, tf, dt, G=6.67430e-11):
    t_list = [t]
    y_list = [y.copy()]
    N = len(masses)

    # Initial Acceleration
    acc = get_acc(y, masses, G)

    while t < tf:
        if t + dt > tf:
            dt = tf - t

        y_matrix = y.reshape((N, 6))
        velocities = y_matrix[:, 3:6]

        # 1. Half-Kick (Velocity)
        velocities += 0.5 * acc.reshape((N, 3)) * dt

        # 2. Drift (Position)
        y_matrix[:, :3] += velocities * dt

        # 3. Update Acceleration (using new positions)
        acc = get_acc(y, masses, G)

        # 4. Half-Kick (Velocity)
        velocities += 0.5 * acc.reshape((N, 3)) * dt

        t += dt
        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)


def rk45(f, t0, y0, tf, dt, tol):
    # set parameters.
    dt_min = 1e-6
    dt_max = dt

    # Initial Time
    t = t0

    y = y0.copy()
    t_list = [t]

    y_list = [y.copy()]

    # do while loop until t reaches tf.
    while t < tf:
        if t + dt > tf:
            dt = tf - t

        # The K1-K6 Slopes.
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 4, y + k1 / 4)
        k3 = dt * f(t + 3 * dt / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = dt * f(
            t + 12 * dt / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197
        )
        k5 = dt * f(
            t + dt, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104
        )
        k6 = dt * f(
            t + dt / 2,
            y
            - 8 * k1 / 27
            + 2 * k2
            - 3544 * k3 / 2565
            + 1859 * k4 / 4104
            - 11 * k5 / 40,
        )

        # The 4-th and 5-th ODE
        y4 = (
            y + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
        )
        y5 = (
            y
            + (16 / 135) * k1
            + (6656 / 12825) * k3
            + (28561 / 56430) * k4
            - (9 / 50) * k5
            + (2 / 55) * k6
        )

        scale = np.maximum(np.abs(y), 1.0)

        error = np.sqrt(np.mean(((y5 - y4) / scale) ** 2))

        error = max(error, 1e-16)

        if error <= tol or dt <= dt_min:
            t += dt
            y = y5
            t_list.append(t)
            y_list.append(y.copy())

        dt = 0.9 * dt * (tol / error) ** 0.2

        dt = min(max(dt, dt_min), dt_max)

    return np.array(t_list), np.array(y_list)


def rk4(f, t, y, tf, dt):
    t_list = [t]
    y_list = [y.copy()]
    while t < tf:
        if t + dt > tf:
            dt = tf - t
        k1 = dt * f(t, y)
        k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1)
        k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2)
        k4 = dt * f(t + dt, y + k3)

        y_new = y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = y_new
        t += dt

        t_list.append(t)
        y_list.append(y_new.copy())
    return np.array(t_list), np.array(y_list)
