# Computational Linear Algebra Ep#11 Line-Fitting
# By: Nick Space Cowboy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import argparse

def polyfit(x_data, y_data, order):
    aorder = order + 1
    b = np.array(y_data)
    A = np.ones((len(b), aorder))
    for i in range(1, aorder):
        A[:,i] = np.array(x_data) ** i
    lsq = np.linalg.lstsq(A, b.T, rcond=None)
    x = lsq[0]
    try: 
        norm = lsq[1][0]**(0.5)
        return [x, norm]
    except IndexError:
        print("WARNING: Potential Overfit")
        mnorm = sum((np.dot(A, x) - b) ** 2) ** (0.5)
        return [x, mnorm]
    
def plot(din, line_params, title, filename):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(sharey=True)
    ax.set_ylabel("y-axis")
    ax.set_xlabel("x-axis")
    x = np.arange(0, din["Total Numbers"].iloc[-1], 1e3)
    y = 0
    for i in range(0, len(line_params[0])):
        y += line_params[0][i] * (x ** i)
    poly, = ax.plot([], [], zorder=2, linewidth=2.0, color="aqua", label="Line Fit")
    
    def animate(n, x, y, line):
        poly.set_data(x[:n], y[:n])
        return line,
        
    fig.suptitle(title)
    ax.plot(din["x"], din["y"], label="Runtime Data", color="orangered", zorder=1)
    plt.legend()
    ani = animation.FuncAnimation(fig, animate, len(x), fargs=[x, y, poly])
    plt.show()
    # ani.save(f"{filename}.mp4", fps=5)
    
def gen_random():
    y = np.random.randint(1, high=25, size=4)
    x = np.arange(1, len(y) + 1)
    assert len(x) == len(y)
    df = pd.DataFrame({"x": x,
                        "y": y})
    print("Data Generated:")
    print(df)
    return df
    
    
def startup():
    p = argparse.ArgumentParser(description="Tool to create animations for basic model fittings")
    p.add_argument("-model",
                    "--model", 
                    help="Type of model to fit to data",
                    default="std-poly",
                    type=str)
    p.add_argument("-mode",
                    "--mode",
                    help="Type of model to fit to data",
                    default="random",
                    type=str)
#    p.add_argument("-show",
#                    "--show",
#                    "show",
#                    help="Type of model to fit to data",
#                    default="all",
#                    type=str)
                    
    modes = ["random", "data"]
    models = ["std-poly"]
    data = 0
    args=p.parse_args()
#    shows = ["none", "all", "plot"]
    if args.mode == "random":
        data = gen_random()

    fit_status = True
    i = 1
    norm = None
    while fit_status == True:
        print(f"{i} order polynomial fit")
        fit = polyfit(data["x"], data["y"], i)
        c = 0
        for j in fit[0]:
            print(f"C{c}: {j}")
            c += 1
        print(f"Norm: {fit[1]}")
        if norm is None or norm > fit[1]:
            norm = fit[1]
        else: 
            fit_status = False
        i += 1
        print("")
        # plot(data, fit, f"{i}th Order Polyomial Rust Runtime Fit", f"rust-runtime-fit{i}")
    
