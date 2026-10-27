import numpy as np
import matplotlib.pyplot  as plt

import math

scan_data = [
    (0,300),
    (45,500),
    (90,123),
    (135,214),
    (180,450),
    (225,96),
    (270,678),
    (315,631)
]

def generate_map(scan_data):
    angles = [math.radians(angle) for angle,radius in scan_data]
    radiuses = [radius for angle,radius in scan_data]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection="polar")
    ax.scatter(angles,radiuses,c='blue',s=20,label="Obstacles(mm)")
    ax.plot(angles+[angles[0]],radiuses+[radiuses[0]],color="red",alpha=0.5)
    ax.set_title("Standard Math Polar Map of Lidar LD19 Obstacle Data",va='bottom')
    ax.grid(True)
    ax.legend(loc='lower right')
    plt.savefig("plot_polar_lidar.png")
    print('Map Saved')
    plt.show()


generate_map(scan_data)