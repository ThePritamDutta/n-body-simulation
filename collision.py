import numpy as np

def collision(y, masses, radii, contact_state, k, eps=1e-6):
    
    N = len(masses)
    e = 1.0  # perfectly elastic, can be <1 for damping

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

            if dist < 1e-12:
                continue

            # check penetration or touch
            if dist < min_dist + eps:
                n = rel_pos / dist
                v_rel = vel2 - vel1
                v_normal = np.dot(v_rel, n)

                # collision resolution only on contact enter
                if not contact_state[i][j]:
                    if v_normal < -eps:   # must be closing
                        m1, m2 = masses[i], masses[j]
                        j_impulse = -(1 + e) * v_normal / (1/m1 + 1/m2)
                        impulse = j_impulse * n

                        # apply velocity impulse
                        y[idx + 3:idx + 6] -= impulse / m1
                        y[jdx + 3:jdx + 6] += impulse / m2 if m2 > 0 else 0

                        # positional separation (with bias)
                        overlap = min_dist - dist
                        corr = (overlap + eps) * n
                        y[idx:idx+3] -= 0.5 * corr
                        y[jdx:jdx+3] += 0.5 * corr

                        k += 1   # collision count

                # mark as contact
                contact_state[i][j] = True

            else:
                # out of contact -> reset event gate
                contact_state[i][j] = False

    return y, contact_state, k

