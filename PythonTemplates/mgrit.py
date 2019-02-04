import numpy as np


def mgrit(l, L, u, v, g, m, t, Phi, app):
    # Solve
    if l == L - 1:
        for i in range(1, np.size(t[l])):
            u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])
        return u, v, g

    # Compute F and C points
    nt = np.size(t[l])
    cpts = np.array(range(0, nt, m))
    fpts = np.array(list(set(np.array(range(0, nt))) - set(cpts)))

    # FCF-relaxation
    for i in np.nditer(fpts):
        u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

    for i in np.nditer(cpts[1:]):
        u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

    for i in np.nditer(fpts):
        u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

    v[l + 1] = np.copy(u[l][cpts])

    for i in range(1, np.size(cpts)):
        g[l + 1][i] = g[l][cpts[i]] \
                      + Phi(u[l][cpts[i] - 1], t[l][cpts[i] - 1], t[l][cpts[i]], app[l]) - u[l][cpts[i]] \
                      - Phi(v[l + 1][i - 1], t[l + 1][i - 1], t[l + 1][i], app[l + 1]) + v[l + 1][i]

    u[l + 1] = np.copy(v[l + 1])

    u, v, g = mgrit(l + 1, L, u, v, g, m, t, Phi, app)

    e = u[l + 1] - v[l + 1]

    u[l][cpts] = u[l][cpts] + e

    # F-relaxation
    for i in np.nditer(fpts):
        u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

    return u, v, g
