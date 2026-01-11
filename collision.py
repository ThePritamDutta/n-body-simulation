import numpy as np

def collision(y, masses, radii, e=0.8):
    G = 6.67e-11
    N = len(masses)

    collision_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            r1 = y[i,:3]; r2 = y[j,:3]
            dist = np.linalg.norm(r2 - r1)
            if dist <= radii[i] + radii[j]:
                collision_pairs.append((i,j))

    if not collision_pairs:
        return y, masses, radii

    removed = set()
    to_add = []

    for (i,j) in collision_pairs:
        if i in removed or j in removed:
            continue

        r1 = y[i,:3]; r2 = y[j,:3]
        v1 = y[i,3:]; v2 = y[j,3:]
        m1 = masses[i]; m2 = masses[j]

        rel = r2 - r1
        dist = np.linalg.norm(rel) + 1e-12
        n = rel/dist
        v_rel = v1 - v2
        v_rel_mag = np.linalg.norm(v_rel)

        v_esc = np.sqrt( 2*G*(m1+m2)/(radii[i]+radii[j]) )

        # ---- MERGE ----
        if v_rel_mag < v_esc:
            m_new = m1+m2
            pos_new = (m1*r1 + m2*r2)/m_new
            vel_new = (m1*v1 + m2*v2)/m_new
            r_new = (radii[i]**3 + radii[j]**3)**(1/3)

            removed.update([i,j])
            to_add.append((pos_new, vel_new, m_new, r_new))
            continue

        # fargmentation
        if v_rel_mag > 2*v_esc:
            removed.update([i,j])
            m_tot = m1+m2
            k = 3
            m_frag = m_tot/k
            r_frag = ((radii[i]**3 + radii[j]**3)/k)**(1/3)

            # center of momentum
            pos_c = (r1+r2)/2
            vel_c = (m1*v1 + m2*v2)/m_tot

            
            for _ in range(k):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                pos_new = pos_c + direction*r_frag*0.2
                vel_new = vel_c + direction*(v_rel_mag*0.2)
                to_add.append((pos_new, vel_new, m_frag, r_frag))
            continue

        # bounce
        m_tot = m1+m2
        J = -(1+e)*np.dot(v_rel,n)/(1/m1 + 1/m2)
        v1 = v1 + (J/m1)*n
        v2 = v2 - (J/m2)*n

        # separate overlap
        pen = (radii[i]+radii[j]) - dist
        r1 -= n * pen*(m2/m_tot)
        r2 += n * pen*(m1/m_tot)

        y[i,:3]=r1; y[j,:3]=r2
        y[i,3:]=v1; y[j,3:]=v2  

    # remove and add the elements in the actual array.
    if removed:
        keep = sorted(set(range(N)) - removed)
        y = y[keep]
        masses = masses[keep]
        radii = radii[keep]

        for pos,vel,m_new,r_new in to_add:
            new = np.zeros(6)
            new[:3]=pos
            new[3:]=vel
            y = np.vstack((y,new))
            masses = np.append(masses,m_new)
            radii = np.append(radii,r_new)

    return y, masses, radii

