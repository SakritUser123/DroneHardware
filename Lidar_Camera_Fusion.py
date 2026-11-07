

import cv2
import numpy as np

import threading
import time
import camera_depth_analysis as camera_module
from LidarWrapperImported import LD19Lidar, LidarPlot

WIDTH = 320 # WIDTH of view on pixels
HEIGHT = 240 # HEIGHT of view on pixels
HFOV = 66.0 # horizontal field of view in degrees
VFOV = 41.0 # vertical field of view in degrees

LIDAR_HEIGHT = 50.8 # 2 inches x 25.4  to convert into mm

fx = WIDTH / ( 2 * (np.tan(np.radians(HFOV / 2)))) # horizontal focal length in pixel units.
fy = HEIGHT / ( 2 * (np.tan(np.radians(VFOV/2)))) # vertical focal length in pixel units.

cx = WIDTH / 2 # horizontal center of image

cy = HEIGHT / 2 # vertical center of image

stop_event = threading.Event()

camera_thread = threading.Thread(  # makes ure that each process runs simueltanously and doesnt wait for the other to finish.
  target = camera_module.run_camera,
  args = (stop_event,),
  daemon = True

)
camera_thread.start()

lidar = LD19Lidar(
  port_name = "/dev/ttyUSB0",
  min_angle = 0,
  max_angle = 359
)
plotter = LidarPlot()
try:
  while True:
    frame = camera_module.latest_frame
    depth = camera_module.latest_depth
  
    if frame is None or depth is None:
      time.sleep(0.01)
      continue
  
  frame = frame.copy()
  
  angles , distances = lidar.run_threaded()
  
  if len(distances) > 0:
    plotter.run_threaded(angles,distances)
  
  for angle, distance in zip(angles,distances): # if distance is not postive or is above 10,000 mm (too far away ) then it will not be used. 
    if distance <= 0 or distance >= 10000:
      continue
    angle_rad = np.radians(angle) # converts into radians because cosine numpy cos fucntion requires radians.
    forward_component_vector = distance * np.cos(angle_rad) # caluculates forward component formula is forward_comp = distance * cosine of angle
    if forward_component_vector <=0: # if forward component is negative then it will not be used.
      continue
    x = int(cx + fx * np.tan(angle_rad)) 
    # cx is cameras horizontal center
    # fx is the horizontal focal length in pixels.
  
    # x_position = (center + focal_length * tan of angle)
  
    y = int(cy - fy*(LIDAR_HEIGHT/ forward))
  
    # cy is cameras vertical center 
    # fy is vertical focal length
    # LIDAR_HEIGHT is lidar height difference to the camera (assumed) 
    # forward is forward distance  to object in mm
    # so far we assume that the scan area of the LIDAR and the Camera are both parallel to the ground and that the scan plane of the LIDAR intersects a part of the viewing region of the Camera.
    if 0 <= x < WIDTH  and 0 <= y < HEIGHT: # checks that the point is inside the image
      cv2.circle(frame, (x,y) , 4, (0,0,255),-1) # draws a read dot in the image where the Lidar detects the object
      cv2.putText(
      frame,
      f"{int(distance)} mm",
      (x + 5, y - 5), # the addition of 5 and subtrcation of 5 is to make sure the text doesnt overlap with the dot and is slight next to it.
      cv2.FONT_HERSHEY_SIMPLEX,
      0.4,
      (0, 0, 255),
      1
      )
  
    depth_display = cv2.normalize(
      depth, None , 0 , 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Camera + LiDAR", frame)
    cv2.imshow("Depth Anything", depth_display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

finally:
  stop_event.set()
  camera_thread.join(timeout=2)
  lidar.shutdown()
  plotter.shutdown()
  cv2.destroyAllWindows()

  
    
