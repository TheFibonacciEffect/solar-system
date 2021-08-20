# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt


# TODO vervollständigen Sie das Template.

#  from ode_solvers import explicit_euler, implicit_euler
#  from ode_solvers import implicit_mid_point, velocity_verlet


def plot_orbits(y, filename):
    """Erstellt einen Plot der Bahnen aller Teilchen.

    y : 4D-Array. Approximative Lösung des N-Körperproblems.
        Achse 0: Zeitschritte
        Achse 1: Ort und Implus/Geschwindigkeit.
        Achse 2: Verschiedene Teilchen.
        Achse 3: Raumdimensionen.

    filename : Name der Datei unterdem der Plot gespeichert wird, ohne Endung.
    """
    for k in range(y.shape[2]):
        plt.plot(y[:, 0, k, 0], y[:, 0, k, 1])

    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()


def measure_runtime(function, *args, **kwargs):
    """Naive measurement of the runtime of a function.

    This idea of only measuring once, only works for rather slow
    functions. Anything above 0.1 seconds should be safe.

    Naturally, there are still sources of error, e.g., the current CPU load. If
    you're running other expensive tasks in parallel the runtime could be off.

    Examples:

        # fx = f(x)
        t_run, fx = measure_runtime(f, x)

        # fx = f(x, foo=42.0)
        t_run, fx = measure_runtime(f, x, foo=42.0)

        # a, b = f(x)
        t_run, (a, b) = measure_runtime(f, x)
    """

    t0 = time.perf_counter()
    fx = function(*args, **kwargs)
    t1 = time.perf_counter()

    return t1 - t0, fx



# TODO vervollständigen Sie das Template.



def three_body_ic():
    n_bodies = 3
    m = np.array([1.0, 1.0, 1.0])

    y0 = np.zeros((2, n_bodies, 3))

    y0[0, 0, :] = np.array([0.97000436, -0.24308753, 0.0])
    y0[1, 0, :] = np.array([0.46620368, 0.43236573, 0.0])

    y0[0, 1, :] = -y0[0, 0, :]
    y0[1, 1, :] = y0[1, 0, :]

    y0[0, 2, :] = 0.0
    y0[1, 2, :] = np.array([-0.93240737, -0.86473146, 0.0])

    return m, y0


def planet_ic():
    msun = 1.00000597682
    qsun = np.array([0, 0, 0])
    vsun = np.array([0, 0, 0])

    mj = 0.00095486104043
    qj = np.array([-3.5023653, -3.8169847, -1.5507963])
    vj = np.array([0.00565429, -0.00412490, -0.00190589])

    ms = 0.000285583733151
    qs = np.array([9.0755314, -3.0458353, -1.6483708])
    vs = np.array([0.00168318, 0.00483525, 0.00192462])

    mu = 0.0000437273164546
    qu = np.array([8.3101420, -16.2901086, -7.2521278])
    vu = np.array([0.00354178, 0.00137102, 0.00055029])

    mn = 0.0000517759138449
    qn = np.array([11.4707666, -25.7294829, -10.8169456])
    vn = np.array([0.00288930, 0.00114527, 0.00039677])

    mp = 7.692307692307693e-09
    qp = np.array([-15.5387357, -25.2225594, -3.1902382])
    vp = np.array([0.00276725, -0.00170702, -0.00136504])

    masses = np.array([msun, mj, ms, mu, mn, mp]).reshape((-1, 1))

    n_bodies = 6
    y0 = np.empty((2, n_bodies, 3))
    for k, q0 in enumerate([qsun, qj, qs, qu, qn, qp]):
        y0[0, k, :] = q0

    for k, (m, v0) in enumerate(zip(masses, [vsun, vj, vs, vu, vn, vp])):
        y0[1, k, :] = m * v0

    return masses, y0



# TODO vervollständigen Sie das Template.

