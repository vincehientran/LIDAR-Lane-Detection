import matplotlib
import numpy as np
import pyproj
import math

def gps_to_ecef(lat, lon, alt):
    rad_lat = lat * (math.pi / 180.0)
    rad_lon = lon * (math.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(rad_lat) * math.sin(rad_lat))

    x = (v + alt) * math.cos(rad_lat) * math.cos(rad_lon)
    y = (v + alt) * math.cos(rad_lat) * math.sin(rad_lon)
    z = (v * (1 - e2) + alt) * math.sin(rad_lat)

    return x, y, z

def run():
    filePoints = open('final_project_point_cloud.fuse','r')
    fileXYZ = open('test.pcd','a')
    for pt in filePoints.readlines():
        lst = pt.split()
        point = gps_to_ecef(float(lst[0]), float(lst[1]), float(lst[2]))
        #fileXYZ.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + lst[-1] + '\n')

    filePoints.close()
    fileXYZ.close()

if __name__ == '__main__':
    run()
