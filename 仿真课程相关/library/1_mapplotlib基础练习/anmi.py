import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

fig,ax = plt.subplot()
xdata, ydata = [],[]
line, = plt.plot([],[],'ro')

def init():
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-1,1)
    return line,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    line.set_data(xdata,ydata)
    return line,
anim = animation.FuncAnimation(fig,update,frames=np.linspace(-np.pi,np.pi,90),interval=100,init_func==init ,blit=True)
plt.show()