import numpy as np
def collision(y, masses, radii, e=1.0):
    N = len(masses)
    y = y.copy()
    e = 0.2

    for i in range(N):
        for j in range(i+1, N):
            idx = 6*i
            jdx = 6*j

            r1 = y[idx:idx+3]
            r2 = y[jdx:jdx+3]
            v1 = y[idx+3:idx+6]
            v2 = y[jdx+3:jdx+6]

            rel = r2 - r1
            dist = np.linalg.norm(rel)
            if dist < 1e-9:   # avoid singularity
                continue

            if dist <= radii[i] + radii[j]:
                n = rel / dist
                m1, m2 = masses[i], masses[j]

                vrel = np.dot(v1 - v2, n)

                if vrel < 0:  # only if approaching
                    J = -(1 + e)*vrel / (1/m1 + 1/m2)
                    v1 += (J/m1)*n
                    v2 -= (J/m2)*n

                    pen = (radii[i] + radii[j]) - dist
                    total_mass = m1 + m2
                    r1 -= n * pen * (m2/total_mass)
                    r2 += n * pen * (m1/total_mass)

                # write back
                y[idx:idx+3] = r1
                y[jdx:jdx+3] = r2
                y[idx+3:idx+6] = v1
            
    return y

