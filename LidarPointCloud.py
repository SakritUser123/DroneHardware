import numpy as np

import open3d as o3d

h , w = depth.shape


fx = w / (2 * np.tan(np.radians(66/2)))

# square coordiantes so vertical focal length equals horizontal focal length.
fy = fx

cx = w/ 2

cy = h/ 2

Z = (depth-depth.min()) / (depth.max()-depth.min() + 1e-8)

Z = 1 - Z

# u contains x pixel positions all
# v contains the y pixel postiions all.

u , v = np.meshgrid(np.arrange(w), np.arrange(h))

X = (u - cx) * Z / fx # calculates 3d horizontol position for each pixel.
Y = (v - cy) * Z / fy # calculates 3d vertical position for each pixel.

#creates all points by combing respectie x and y and z points it also reverses the y coordinates in order to match the chosen convention.
points = np.stack((X, -Y, Z), axis=-1).reshape(-1, 3) # creates each point in to a array [x,y,z]





