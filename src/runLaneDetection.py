import numpy as np
import math
import pptk

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

    # plotting the points
    pts = []
    for pt in filePoints.readlines():
        lst = pt.split()
        point = gps_to_ecef(float(lst[0]), float(lst[1]), float(lst[2]))
        #fileXYZ.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + lst[-1] + '\n')
        intensity = int(lst[-1])
        if intensity > 20:
            pts.append([point[0], point[1], point[2], 255, 0, 0])
        else:
            pts.append([point[0], point[1], point[2], 0, 0, 0])


    pts = np.array(pts)
    xyz = pts[:,:3]
    rgb = pts[:,3:]
    v = pptk.viewer(xyz, rgb)
    v.set(point_size=0.05)

    print('Start finding road plane.')
    road_points(pts)

    filePoints.close()
    fileXYZ.close()

def road_points(pts):

    threshold = 0.2
    bestPlane = None
    bestPlaneFitPoints = 0

    for _ in range(50):

        # find the plane equation
        points = pts[np.random.randint(pts.shape[0], size=3), :]
        plane = planeEquation(points)

        fitPoints = 0
        for pt in pts[::5]:
            distance = distance_point_plane(plane, pt)
            if distance < threshold:
                fitPoints += 1

        if fitPoints > bestPlaneFitPoints:
            bestPlane = plane
            bestPlaneFitPoints = fitPoints

    roadPts = []
    for pt in pts:
        distance = distance_point_plane(bestPlane, pt)
        if distance < threshold:
            roadPts.append(pt)

    roadPts = np.array(roadPts)
    xyz = roadPts[:,:3]
    rgb = roadPts[:,3:]
    v = pptk.viewer(xyz, rgb)
    v.set(point_size=0.05)


def planeEquation(points):
    vector1 = points[0,:3] - points[1,:3]
    vector2 = points[1,:3] - points[2,:3]
    normal = np.cross(vector1, vector2)
    x0 = points[0,0]
    y0 = points[0,1]
    z0 = points[0,2]
    d = (normal[0] * (-x0)) + (normal[1] * (-y0)) + (normal[2] * (-z0))
    plane = np.array([normal[0], normal[1], normal[2], d])

    return plane

def distance_point_plane(plane, point):
    numerator = abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3])
    denominator = math.sqrt((plane[0]**2) + (plane[1]**2) + (plane[2]**2))

    return numerator/denominator



if __name__ == '__main__':
    run()
