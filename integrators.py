import numpy as np
import collison

G=6.67430e-11

def get_acc(state, masses, N, G=6.67430e-11, eps=1e4):
    N = len(masses)
    acc = np.zeros((N, 3))
    positions = state.reshape((N, 6))[:, :3]

    for i in range(N):
        xi, yi, zi = positions[i]

        for j in range(i + 1, N):
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi

            dist2 = dx*dx + dy*dy + dz*dz + eps*eps
            inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))

            factor = G * masses[i] * masses[j] * inv_dist3

            fx = dx * factor
            fy = dy * factor
            fz = dz * factor

            acc[i, 0] += fx / masses[i]
            acc[i, 1] += fy / masses[i]
            acc[i, 2] += fz / masses[i]

            acc[j, 0] -= fx / masses[j]
            acc[j, 1] -= fy / masses[j]
            acc[j, 2] -= fz / masses[j]

    return acc.ravel()


def verlet_step(t, y, masses, tf, dt, radii):
    t_list = [t]
    y_list = [y.copy()]
    N = len(masses)

    # initial acceleration
    acc = get_acc(y, masses, N, G, 1e4)

    while t < tf:
        dt_local = min(dt, tf - t)   # prevents decreasing dt permanently

        # ---- HALF KICK ----
        for i in range(N):
            y[6*i+3:6*i+6] += 0.5 * acc[i] * dt_local   # v += a*dt/2

        # ---- DRIFT ----
        for i in range(N):
            y[6*i:6*i+3] += y[6*i+3:6*i+6] * dt_local   # x += v*dt

        # ---- COLLISION ----
        y = collision.collision(y, masses, radii, e=0.8)

        # ---- NEW ACCEL ----
        acc = get_acc(y, masses, N, G, 1e4)

        # ---- SECOND HALF KICK ----
        for i in range(N):
            y[6*i+3:6*i+6] += 0.5 * acc[i] * dt_local   # v += a*dt/2

        t += dt_local
        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)



def rk45(f, t0, y0, tf, dt, tol,masses,radii):
    # set parameters.
    dt_min = 1e-6
    dt_max = 3e3

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
            y = collision.collision(y, masses, radii)
            y_list.append(y.copy())
            

        dt = 0.9 * dt * (tol / error) ** 0.2

        dt = min(max(dt, dt_min), dt_max)

    return np.array(t_list), np.array(y_list)


def rk4(f, t, y, tf, dt,masses,radii):
    t_list = [t]
    y_list = [y.copy()]

     # do while loop until t reaches tf.
    while t < tf:
        if t + dt > tf:
            dt = tf - t
            
          # The K1-K4 Slopes.
        k1 = dt * f(t, y)
        k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1)
        k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2)
        k4 = dt * f(t + dt, y + k3)
        
        # The ODE
        y_old = y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_new = collision.collision(y_old, masses, radii)
        
        y = y_new
        t += dt

        t_list.append(t)
        y_list.append(y_new.copy())
    return np.array(t_list), np.array(y_list)


