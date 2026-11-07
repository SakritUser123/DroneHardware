import time
from lidar_wrapper  import LD19Lidar,LidarPlot
def main():
    lidar_out = LD19Lidar(port_name='/dev/ttyUSB0',min_angle=0,max_angle=359)
    plotter = LidarPlot()

    try:
        while True:
            angles, distances = lidar_out.run_threaded()
            left, center , right, back,decision = lidar_out.get_sector_readings(angles,distances)
           
            print("LEFT: " , left)
            print("CENTER: " , center)
           
            print("RIGHT: " , right)
            print ("BACK: " , back)
            print("DECISION: " , decision)
           
            plotter.run_threaded(angles,distances)
           
           
           
    except KeyboardInterrupt:
        lidar_out.shutdown()
        plotter.shutdown()

if __name__ == "__main__":
    main()
