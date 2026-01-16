import numpy as np

def collision(y, masses, radii, e=1.0):
    N = len(masses)
    
    for i in range(N):
        for j in range(i + 1, N):
            idx = 6 * i
            jdx = 6 * j

            pos1 = y[idx : idx + 3]
            pos2 = y[jdx : jdx + 3]
            vel1 = y[idx + 3 : idx + 6]
            vel2 = y[jdx + 3 : jdx + 6]

            rel_pos = pos2 - pos1
            dist = np.linalg.norm(rel_pos)
            min_dist = radii[i] + radii[j]

            # ---- SAFETY GUARD ----
            if dist < 1e-12:
                continue

            if dist < min_dist:
                n = rel_pos / dist
                v_rel = vel2 - vel1
                v_normal = np.dot(v_rel, n)

                if v_normal < 0:
                    m1, m2 = masses[i], masses[j]
                    j_impulse = -(1 + e) * v_normal / (1/m1 + 1/m2)
                    impulse = j_impulse * n

                    y[idx + 3 : idx + 6] -= impulse / m1
                    y[jdx + 3 : jdx + 6] += impulse / m2

                    # Position correction
                    overlap = min_dist - dist
                    correction = overlap * n
                    y[idx:idx+3] -= 0.5 * correction
                    y[jdx:jdx+3] += 0.5 * correction


    return y
