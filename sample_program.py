import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def sample_vector(x, y):
    
    vx = x**2 + x*y - 2*x
    vy = x*y**2 + 3*y

    return vx, vy


def diff_func(t,z):
    x, y = z
    dx = x**2 + x*y - 2*x
    dy = x*y**2 + 3*y
    return [dx,dy]


def grid_options(start, stop, stepsize, numpoints, option):
    
    if option == "arange":
        # arange creates a set of values between a start point and a stop point in steps of stepsize.
        x, y = np.meshgrid(np.arange(start, stop, stepsize), np.arange(start, stop, stepsize))
    elif option == "linspace":
        # linspace creates a set of numpoints values between a start point and a stop point
        # start point and stop point are both always included
        x, y = np.meshgrid(np.linspace(start, stop, numpoints), np.linspace(start, stop, numpoints))
    else:
        raise NameError("Invalid option {} selected.".format(option))
    return x, y


def reduced_quiverplot(x, y, vx, vy, label, key, reduction, x_pos=0.9, y_pos=0.9, key_size=2):
    
    plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square

    q = plt.quiver(x[::reduction, ::reduction], y[::reduction, ::reduction],  # coordinates at reduced density
                   vx[::reduction, ::reduction], vy[::reduction, ::reduction],  # arrow x/y lengths at reduced density
                   pivot='mid',  # position of the pivot of the arrow
                   label=f'{label}')  # label using LaTex notation

    # creates the quiver key as above in the position specified
    if key == True:
        plt.quiverkey(q, x_pos, y_pos, key_size, label, labelpos='E', coordinates='figure')
    plt.legend()

    return q


def main():
    
    x, y = grid_options(-5, 5, 0.1, 101, "linspace")
    vx, vy = sample_vector(x, y)
    initial_conditions = [
        (-0.2, 0.4),
        ( 2.4, 0.2),
        ( 2.9,  -2),
        (-0.6, 3.1)
    ]
    t = np.linspace(0,1.5,101,endpoint=True)

    # Define curves for nullclines
    x_full = np.linspace(-5, 5, 400)
    x_pos = x_full[x_full > 0]
    x_neg = x_full[x_full < 0]

    y_vx1 = 2 - x_full       # y = 2 - x
    y_vy2_pos = -3 / x_pos   # y = -3/x
    y_vy2_neg = -3 / x_neg

    # Fixed Points
    fixed_points = np.array([[0,0],[2,0],[3,-1],[-1,3]])

    # Plot setup
    plt.figure(figsize=(7, 7))
    plt.gca().set_aspect('equal', adjustable='box')

    # Vector field
    reduced_quiverplot(x, y, vx, vy,label= "i",key = False, reduction=5)
  
    # trivial nullclines (x , y axes)
    plt.axvline(0, color='k', linewidth=1.2, label='x, y  = 0')
    plt.axhline(0, color='k', linewidth=1.2)
    
    # nullclines (non-trivial)
    plt.plot(x_full, y_vx1, 'b--', label='y = 2-x')
    plt.plot(x_pos, y_vy2_pos, 'r--', label='y = -3/x')
    plt.plot(x_neg, y_vy2_neg, 'r--')

    # plot fixed points
    plt.scatter(fixed_points[:, 0], fixed_points[:, 1],
    color='k', s=20, zorder=5, label='Fixed points')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    for (x0, y0), point in zip(initial_conditions,fixed_points):
        result = integrate.solve_ivp(diff_func,(0,1.5),y0 = [x0,y0],method="RK45",t_eval=t)
        x,y = result.y
        t_eval = result.t
        plt.plot(t_eval, x, color = "r")
        plt.plot(t_eval, y, color = "b")
        plt.axhline(y=point[0],color = "k", linestyle = "--")
        plt.axhline(y=point[1],color = "k", linestyle = "--")
        plt.show()


if __name__ == "__main__":
    main()
