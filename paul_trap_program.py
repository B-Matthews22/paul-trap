import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sample_program as sp


def paul_vector(t, x, y, a, b):
    
    vx = y
    vy = (b * np.cos(2*t) - a) * x

    return vx, vy

def paul_ode(t, z, a, b):

    x, y = z
    dx = y
    dy = (b * np.cos(2*t) - a) * x

    return dx, dy 

def main():

    a = -2
    b = 2
    t = np.pi*0.5

    x, y   = sp.grid_options(-5, 5, 5, 101, "linspace")
    vx, vy = paul_vector(t, x, y, a, b)

    sp.reduced_quiverplot(x, y, vx, vy, label = "p", key = False, reduction = 10)
    plt.show()

if __name__ == "__main__":
    main()