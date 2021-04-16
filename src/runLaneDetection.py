import numpy as np
import math
import pptk
from sklearn.cluster import DBSCAN

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
    #display(pts)

    print('Find road plane.')
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
    #display(roadPts)
    lane_candidates(roadPts, pts)

def lane_candidates(roadPts, all_pts):
    points = []
    for pt in roadPts:
        # if the pt R value is not 0
        if pt[3] > 0:
            points.append(pt)
    points = np.array(points)
    #display(points)

    lane_points(points, all_pts)

def lane_points(pts, all_pts):
    print('Find lane lines.')
    threshold = 0.3

    lane_lines = []
    remaining_pts = np.copy(pts)
    total_pts = len(pts)

    # if each lane has more than 2% of the total points, continue finding lanes
    counter = 0
    bestLineFitPoints = total_pts
    while bestLineFitPoints > total_pts*0.02:
        bestLine = None
        bestLineFitPoints = 0
        for _ in range(200):
            # find the line equation
            points = remaining_pts[np.random.randint(remaining_pts.shape[0], size=2), :]
            line = lineEquation(points)

            fitPoints = 0
            for pt in remaining_pts[::5]:
                distance = distance_point_line(line, pt)
                if distance < threshold:
                    fitPoints += 1

            if fitPoints > bestLineFitPoints:
                bestLine = line
                bestLineFitPoints = fitPoints

        lane_lines.append(bestLine)
        counter += 1
        print('Found',counter,'line(s).')

        p = []
        for pt in remaining_pts:
            distance = distance_point_line(bestLine, pt)
            if distance > threshold:
                p.append(pt)

        remaining_pts = np.array(p)

    # extract the points that belong to the lane markings
    lane_pts = []
    for line in lane_lines:
        for pt in pts:
            distance = distance_point_line(line, pt)
            if distance < threshold:
                lane_pts.append(pt)
    lane_pts = np.array(lane_pts)
    subset_pts = all_pts[:]
    lane_line_points = generate_lines_points(lane_lines)
    result = np.concatenate((subset_pts, lane_line_points), axis=0)
    display(result)

def generate_lines_points(lines):
    lane_line_points = []
    for line in lines:
        unit_directing_vector = line[3:] / np.linalg.norm(line[3:])
        point_vector = line[:3]
        for i in range(-300, 300):
            point = point_vector + (i * unit_directing_vector)
            point = [point[0], point[1], point[2], 0, 255, 0]
            lane_line_points.append(point)
    lane_line_points = np.array(lane_line_points)
    return lane_line_points

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

def lineEquation(points):
    line = [points[0,0], points[0,1], points[0,2], points[1,0]-points[0,0], points[1,1]-points[0,1], points[1,2]-points[0,2]]
    line = np.array(line)
    return line

def distance_point_plane(plane, point):
    numerator = abs((plane[0] * point[0]) + (plane[1] * point[1]) + (plane[2] * point[2]) + plane[3])
    denominator = math.sqrt((plane[0]**2) + (plane[1]**2) + (plane[2]**2))

    return numerator/denominator

def distance_point_line(line, point):
    directing_vector = line[3:]
    point_vector = line[:3]
    vector_diff = point_vector - point[:3]
    numerator = np.linalg.norm(np.cross(vector_diff, directing_vector))
    denominator = np.linalg.norm(directing_vector)

    return numerator/denominator

def display(points):
    xyz = points[:,:3]
    rgb = points[:,3:]
    v = pptk.viewer(xyz, rgb)
    v.set(point_size=0.05)

if __name__ == '__main__':
    run()
