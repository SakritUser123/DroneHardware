import time
from LidarLD19  import LD19Lidar,LidarPlot
def main():
    lidar_out = LD19Lidar(port_name='/dev/cu.usbserial-0001',min_angle=0,max_angle=359) 
    plotter = LidarPlot()

    try:
        while True:
            normalized_angles, normalized_distances = lidar_out.run_threaded()
            for j in range(len(normalized_distances)):
                if normalized_distances[j] <= 400:
                    print('TOO CLOSE')
                    # REPLACE PART ABOVE WITH PID CONTOLLER MAKING SURE ITS STOPS IF ANY PRINT TOO CLOSE COMES
            # Plot the normalized data
            print(normalized_angles,normalized_distances)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        lidar_out.shutdown()
        plotter.shutdown()

if __name__ == "__main__":
    main()
