import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from nbody import planet_ic
import mpl_toolkits.mplot3d.axes3d as plt3d_ax
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

try: from tqdm import tqdm
except ImportError:
    print("tqdm not found, install tqdm for an awesome progress bar :D")
    def tqdm(x):
        return x
# %%

def right_hand_side(ic,ms,N):
    assert len(ic)%(2*N) == 0, f"the number of planets {N} doesnt match the input with shape {np.shape(ic)}"
    dim = 3
    state = np.reshape(ic, (2*N, dim)) # reshape so that it is a vector of momenta and positions
    ps = state[N:]
    qs = state[:N]
    q_dot = ps/ms.reshape(-1,1)
    p_dot = G*get_sum(qs,ms)
    # return the flattened output with [pos| momentum]
    return np.append(q_dot.reshape(-1), p_dot.reshape(-1))

def acc_vvv(positions,ms,N):
    assert len(positions)%N == 0, f"the number of planets {N} doesnt match the input with shape {np.shape(positions)}, ie {np.shape(positions)}%{N} != 0"
    dim = 3
    state = np.reshape(positions, (N, dim)) # reshape so that it is a vector of positions
    F = G*get_sum(state,ms)# p_dot/m = p_double_dot
    M = ms.reshape(-1,1)
    F = F.reshape(N, dim)
    q_dd = F/M
    return np.ravel(q_dd)

def get_sum(qs, ms): # (6, 3), (6, 1)
    out = np.zeros_like(qs, dtype=np.float32)
    for k,qk in enumerate(qs):
        for i, qi in enumerate(qs):
            if i == k:
                continue
            out[k] += ms[i]*ms[k]/np.linalg.norm(qi-qk)**3 * (qi - qk)
    return np.ravel(out)

def impl_mpr(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in tqdm(range(num-1)):
        sol[i+1] = fsolve(lambda x: x - (sol[i] + dt*right_hand_side(0.5*(sol[i] + x), ms, n)) ,sol[i])
    return sol

def expl_euler(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in tqdm(range(num-1)):
        sol[i+1] = sol[i] + dt*right_hand_side(sol[i], ms, n)
    return sol

def vvv(T,num, acc, initial_conditions, ms,n):
    """returns: positions with shape: (num, N*3)
    """
    t, dt = np.linspace(0,T, num, retstep=True)
    xi = initial_conditions[:N*3]
    pi = initial_conditions[3*N:]
    ms = np.ravel(ms)
    pi =[v/ms[n//3] for n,v in enumerate(pi) ]
    x = np.empty((num,) + np.shape(xi))
    v = np.empty((num,) + np.shape(pi))
    x[0] = xi
    v[0] = pi
    for i in tqdm(range(num-1)):
        x[i+1] = x[i] + dt* v[i] + 0.5* dt**2*acc(x[i], ms, n)
        v[i+1] = v[i] + dt/2* (acc(x[i], ms, n) + acc(x[i+1], ms, n))
    # NOTE Velvet only returns the positions, not the momenta
    return x

def plot_orbits(three_dim, y, animate):
    if animate:
        plot_animation(y)
        return

    N = len(y[0])/(3*2)
    N = int(N)
    fig = plt.figure()
    if three_dim:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if method == "vvv":
        N = int(len(y[0])/(3))

    y_t = np.transpose(y)
    slicing = 1
    for i in range(N):
        xi = y_t[3*i][::slicing]
        yi = y_t[3*i+1][::slicing]
        zi = y_t[3*i+2][::slicing]
        if three_dim:
            ax.plot(xi,yi,zi, "-")
        else:
            plt.plot(xi,yi)
    plt.show()
    return

def plot_animation(data):
    number_of_planets = data.shape[1]//3 if velocity_velvet else data.shape[1]//6
    speed = int(input("animation speed \n(6 works good for 2000 datapoints in the case of the solar system, this needs to be adjusted, when number of points change, eg. use 12 for 4000 datapoints) \ndefault is 6: ") or 6)

    plt.style.use('dark_background')
    plt.rcParams['grid.color'] = "dimgray"
    def update_planets(num, dataLines, lines) :
        for line, data in zip(lines, dataLines) :
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2,num])
        return lines

    def update_lines(num, dataLines, lines) :
        for line, data in zip(lines, dataLines) :
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2,:num])
        return lines

    def update(num, dataLines, lines):
        n = int(len(lines)/2)
        out = update_planets(num, dataLines, lines[:n])  + update_lines(num, dataLines, lines[n:])
        return out

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = plt3d_ax.Axes3D(fig)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    # passing first data points
        
    data = np.transpose(data) # [time, posistions | momenta] -> [posistions | momenta, time]
    data = np.array([[data[3*n,:], data[3*n+1,:], data[3*n +2, :]] for n in range(number_of_planets)])
    labels = ["sun","jupyter", "saturn", "uranus", "neptune", "pluto"]
    colors = ["yellow", "darkorange", "orange","cornflowerblue", "darkblue", "dimgray"]
    planets = [ax.plot(planet[0,0], planet[1,0], planet[2,0], "o", color= color, label=label)[0] for planet, color, label in zip(data, colors, labels)]
    lines = [ax.plot(planet[0,0], planet[1,0], planet[2,0], "--", color= color)[0] for planet, color, label in zip(data, colors, labels)]

    # Setting the axes properties
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))

    ax.set_xlim3d([-15, 20.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-30.0, 25.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-15.0, 5.0])
    ax.set_zlabel('Z')

    ax.set_title('our solar sytem')

    # Creating the Animation object
    planets.extend(lines)
    number_of_time_points = data.shape[-1]
    line_ani = FuncAnimation(fig, update, np.arange(0,number_of_time_points,speed), fargs=(data, planets), interval=50, blit=True)
    # line_ani = FuncAnimation(fig, update_lines, np.arange(0,number_of_time_points,1), fargs=(data, lines), interval=50, blit=True)
    plt.legend()
    if "y" == input("do you want to save the animation? (y/n)"):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        line_ani.save('im.mp4', writer=writer)
    plt.show()

if __name__== "__main__":
    d = {"y" : True,"n": False}
    task = input("which task should be solved? (d,e,f) (d: two body problem, e: Stable solution of the 3 Body problem, f: Model of the solar system) \ntype something else to enter your own input ")
    if task == "d":
        N = 2
        G = 1
        m1 = 500
        m2 = 1
        q1 = np.array([0,0,0])
        q2 = np.array([2,0,0])
        p1 = [0,0,0]
        p2 = [0,np.sqrt(G*m1*0.5),0]
        ic = np.ravel([q1,q2,p1,p2])
        ms = np.array([m1,m2])
    elif task=="e":
        N = 3
        G = 1
        q1 = np.array([0.97000436 + 0.01,-0.24308753 ,0])
        q2 = np.array((-0.97000436,0.24308753,0))
        q3 = np.zeros(3)
        p1 = (0.46620368,0.43236573,0)
        p2 = (0.46620368,0.43236573,0)
        p3 = (-0.93240737,-0.86473146,0)
        ic = np.ravel([q1,q2, q3,p1,p2, p3])
        ms = np.array([1,1,1])
    elif task=="f":
        N = 6
        G = 2.95912208286e-4
        ms, ic = planet_ic()
        ic = np.ravel(ic)
    else:
        own = d[input("do you want to enter your own input? (y/n)") or "n"]
        if own:
            N = int(input("number of planets: "))
            G = float(input("G: "))
            q = []
            p = []
            ms = []
            for i in range(N):
                print(f"enter data for planet {i}")
                m = float(input("mass= "))
                assert m > 0, "there is no negative mass"
                qx = float(input("qx: ") or 0)
                qy = float(input("qy: ") or 0)
                qz = float(input("qz: ") or 0)
                px = float(input("px: ") or 0)
                py = float(input("py: ") or 0)
                pz = float(input("pz: ") or 0)
                q.extend([qx,qy,qz])
                p.extend([px,py,pz])
                ms.append(m)
            ic = q + p
        else:
            print("okay, the program will now terminate")
            quit()

    method = input("""method (expl_euler, mpr, vvv): (default: Velocity Verlet) \n Verlet is the fastest method, the other ones are the explicit euler (does not conserve energy) and 
                   the implicit midpoint rule, wich does conserve energy but is really slow""") or "vvv"
    translator = {
        "expl_euler": expl_euler,
        "mpr": impl_mpr,
        "vvv": vvv
    }
    T = float(input("endtime (default 2e4): ") or "2e4")
    num = int(float(input("steps (default 2000): ") or 2e3))
    output = d[input("3D output? (y/n) (default: y)") or "y"]
    animate = d[input("animate? (y/n) (default y): ") or "y"]

    if method == "vvv":
        y = translator[method](T,num,acc_vvv,ic, ms, N)
        velocity_velvet = True
    else: 
        velocity_velvet = False
        y = translator[method](T,num,right_hand_side,ic, ms, N)
    plot_orbits(output, y, animate)
