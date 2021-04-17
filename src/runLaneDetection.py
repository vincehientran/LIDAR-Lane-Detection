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

def ecef_enu(x, y, z, lat0, lon0, h0):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2-f)

    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp

def run():
    filePoints = open('final_project_point_cloud.fuse','r')

    # convert lla to ecef
    ecef_pts = []
    for pt in filePoints.readlines():
        lst = pt.split()
        point = gps_to_ecef(float(lst[0]), float(lst[1]), float(lst[2]))
        intensity = int(lst[-1])
        ecef_pts.append([point[0], point[1], point[2], float(lst[0]), float(lst[1]), float(lst[2]), intensity])

    # convert ecef to enu
    enu_pts = []
    # supposed to be origin of point cloud, but earth is so big it doesnt matter much
    center = ecef_pts[0]
    lat0, lon0, alt0 = center[3:-1]
    for pt in ecef_pts:
        x, y, z = pt[:3]
        intensity = pt[-1]
        point = ecef_enu(x, y, z, lat0, lon0, alt0)
        # x, y, z, (r,g,b for lane detection), (r,g,b for intensity), alpha value for plane, alpha value for lane candidates, alpha value for lane points
        enu_pts.append([point[0], point[1], point[2], intensity*10, intensity*10, intensity*10, intensity*10, intensity*10, intensity*10, 255, 255, 255])

    enu_pts = np.array(enu_pts)

    print('all points:',len(enu_pts))
    print('Find road plane.')
    road_points(enu_pts)

    filePoints.close()

def road_points(pts):
    threshold = 0.22
    bestPlane = None
    bestPlaneFitPoints = 0

    for _ in range(300):

        # find the plane equation
        points = pts[np.random.randint(pts.shape[0], size=3), :]
        plane = planeEquation(points)

        fitPoints = 0
        for pt in pts[::150]:
            distance = distance_point_plane(plane, pt)
            if distance < threshold:
                fitPoints += 1

        if fitPoints > bestPlaneFitPoints:
            bestPlane = plane
            bestPlaneFitPoints = fitPoints

    plane_points = []
    non_plane_points = []
    for pt in pts:
        distance = distance_point_plane(bestPlane, pt)
        if distance < threshold:
            plane_points.append(pt)
        else:
            non_plane_points.append([pt[0], pt[1], pt[2], 255, 255, 255, pt[6], pt[7], pt[8], 0, 0, 0])

    plane_points = np.array(plane_points)
    non_plane_points = np.array(non_plane_points)
    #display(plane_points)
    print('plane points:',len(plane_points))
    print('non plane points:',len(non_plane_points))
    lane_candidates(plane_points, pts, non_plane_points)

def lane_candidates(plane_points, all_pts, non_plane_points):
    lane_candidate_points = []
    non_lane_plane_points = []
    for pt in plane_points:
        # if the pt R value is not 0
        if pt[3] > 200:
            lane_candidate_points.append(pt)
        else:
            non_lane_plane_points.append([pt[0], pt[1], pt[2], 20, 75, 20, pt[6], pt[7], pt[8], pt[9], 0, 0])
    lane_candidate_points = np.array(lane_candidate_points)
    non_lane_plane_points = np.array(non_lane_plane_points)
    #display(non_lane_plane_points)
    print('lane candidate points:',len(lane_candidate_points))
    print('non lane plane points:',len(non_lane_plane_points))
    lane_points(lane_candidate_points, all_pts, non_plane_points, non_lane_plane_points)

def lane_points(lane_candidate_points, all_pts, non_plane_points, non_lane_plane_points):
    print('Find lane lines.')
    threshold = 0.3

    lane_lines = []
    remaining_pts = np.copy(lane_candidate_points)
    total_pts = len(lane_candidate_points)

    # if each lane has more than 0.25% of the total points, continue finding lanes
    counter = 0
    bestLineFitPoints = total_pts
    while True:
        bestLine = None
        bestLineFitPoints = 0
        for _ in range(400+(counter*20)):
            # find the line equation
            if len(remaining_pts) < 2:
                print('Less than 2 remaining points left. No lines can be made.')
                break
            points = remaining_pts[np.random.randint(remaining_pts.shape[0], size=2), :]
            line = lineEquation(points)

            fitPoints = 0
            # each time a line is found, look at more points
            for pt in remaining_pts[::max(int(20-(counter*2)),8)]:
                distance = distance_point_line(line, pt)
                if distance < threshold:
                    fitPoints += 1

            if fitPoints > bestLineFitPoints:
                bestLine = line
                bestLineFitPoints = fitPoints

        if bestLineFitPoints < 25:
            print('Last line fitted',bestLineFitPoints,'points.')
            break

        lane_lines.append(bestLine)
        counter += 1
        print('Found',counter,'lane line(s). (',bestLineFitPoints,'points fitted )')

        p = []
        for pt in remaining_pts:
            distance = distance_point_line(bestLine, pt)
            if distance > threshold:
                p.append(pt)

        remaining_pts = np.array(p)

    # extract the points that belong to the lane markings
    non_lane_points = []
    lane_points = []
    remaining_pts = np.copy(lane_candidate_points)
    for line in lane_lines:
        new_remaining = []
        for pt in remaining_pts:
            distance = distance_point_line(line, pt)
            if distance > threshold:
                new_remaining.append(pt)
            else:
                lane_points.append([pt[0], pt[1], pt[2], 255, 0, 255, pt[6], pt[7], pt[8], pt[9], pt[10], pt[11]])
        remaining_pts = np.array(new_remaining)
    lane_points = np.array(lane_points)
    #display(lane_points)
    non_lane_points = []
    for pt in remaining_pts:
        non_lane_points.append([pt[0], pt[1], pt[2], 255, 255, 255, pt[6], pt[7], pt[8], pt[9], pt[10], 0])
    non_lane_points = np.array(non_lane_points)

    lane_line_points = generate_lines_points(lane_lines)
    print('lane points:',len(lane_points))
    print('non lane points:',len(non_lane_points))
    guardrail_points, noise = guardrails(non_plane_points)
    pole_points, noise = pole(noise)
    display_result(guardrail_points, noise , non_lane_plane_points, non_lane_points, lane_points, lane_line_points, pole_points)

def guardrails(non_plane_points):
    threshold = 1

    guard_lines = []
    remaining_pts = np.copy(non_plane_points)

    # find two guardrails
    for counter in range(2):
        bestLine = None
        bestLineFitPoints = 0
        for _ in range(300):
            # find the line equation
            points = remaining_pts[np.random.randint(remaining_pts.shape[0], size=2), :]
            line = lineEquation(points)

            fitPoints = 0
            for pt in remaining_pts[::150]:
                distance = distance_point_line(line, pt)
                if distance < threshold:
                    fitPoints += 1

            if fitPoints > bestLineFitPoints:
                bestLine = line
                bestLineFitPoints = fitPoints

        guard_lines.append(bestLine)
        print('Found',counter+1,'guard rail(s).')

        p = []
        for pt in remaining_pts:
            distance = distance_point_line(bestLine, pt)
            if distance > threshold:
                p.append(pt)

        remaining_pts = np.array(p)

    # extract the points that belong to the guardrails
    guardrail_points = []
    remaining_pts = np.copy(non_plane_points)
    for line in guard_lines:
        new_remaining = []
        for pt in remaining_pts:
            distance = distance_point_line(line, pt)
            if distance > threshold:
                new_remaining.append(pt)
            else:
                guardrail_points.append([pt[0], pt[1], pt[2], 255, 0, 0, pt[6], pt[7], pt[8], 0, 0, 0])
        remaining_pts = np.array(new_remaining)
    guardrail_points = np.array(guardrail_points)
    #display(lane_points)
    noise = []
    for pt in remaining_pts:
        noise.append([pt[0], pt[1], pt[2], 255, 255, 255, pt[6], pt[7], pt[8], 0, 0, 0])
    noise = np.array(noise)

    return guardrail_points, noise

def pole(points):
    min_x = np.min(points[:,0])
    min_y = np.min(points[:,1])
    max_x = np.max(points[:,0])
    max_y = np.max(points[:,1])

    width = int((max_x - min_x) * 10)
    height = int((max_y - min_y) * 10)

    votes = np.zeros((height+1,width+1))

    for pt in points:
        x, y = pt[:2]
        x, y = x - min_x, y - min_y
        votes[int(y*10)][int(x*10)] += 1

    pole_lines = []
    for _ in range(4):
        max_vote = 0
        max_vote_location = (0, 0)

        for i in range(len(votes)):
            for j in range(len(votes[0])):
                if votes[i][j] > max_vote:
                    max_vote = votes[i][j]
                    max_vote_location = (i,j)

        votes[max_vote_location[0]][max_vote_location[1]] = 0
        pole_lines.append([(max_vote_location[1]/10) + min_x, (max_vote_location[0]/10) + min_y, 0, 0, 0, 1])
    print('Found',len(pole_lines),'poles.')

    # extract the points that belong to the poles
    pole_points = []
    remaining_pts = np.copy(points)
    for line in pole_lines:
        new_remaining = []
        for pt in remaining_pts:
            distance = distance_point_line(line, pt)
            if distance > 0.2:
                new_remaining.append(pt)
            else:
                pole_points.append([pt[0], pt[1], pt[2], 100, 100, 255, pt[6], pt[7], pt[8], 0, 0, 0])
        remaining_pts = np.array(new_remaining)
    pole_points = np.array(pole_points)
    #display(lane_points)
    noise = []
    for pt in remaining_pts:
        noise.append([pt[0], pt[1], pt[2], 255, 255, 255, pt[6], pt[7], pt[8], 0, 0, 0])
    noise = np.array(noise)

    return pole_points, noise

def display_result(guardrail_points, noise, non_lane_plane_points, non_lane_points, lane_points, lane_line_points, pole_points):
    result = np.concatenate((guardrail_points, noise), axis=0)
    result = np.concatenate((result, non_lane_plane_points), axis=0)
    result = np.concatenate((result, non_lane_points), axis=0)
    result = np.concatenate((result, lane_points), axis=0)
    result = np.concatenate((result, lane_line_points), axis=0)
    result = np.concatenate((result, pole_points), axis=0)
    display(result)

def generate_lines_points(lines):
    lane_line_points = []
    for line in lines:
        unit_directing_vector = line[3:] / np.linalg.norm(line[3:])
        point_vector = line[:3]
        for i in range(-3000, 3000):
            point = point_vector + ((i/20) * unit_directing_vector)
            point = [point[0], point[1], point[2], 0, 255, 0, 0, 0, 0, 0, 0, 255]
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
    numerator_vector = np.cross(vector_diff, directing_vector)
    numerator = math.sqrt((numerator_vector[0]**2) + (numerator_vector[1]**2) + (numerator_vector[2]**2))
    denominator = math.sqrt((directing_vector[0]**2) + (directing_vector[1]**2) + (directing_vector[2]**2))

    if denominator == 0:
        return 100

    return numerator/denominator

def display(points):
    xyz = points[:,:3]
    rgb = points[:,3:6]/255
    intensity_attribute = points[:,6:9]/255
    plane_attribute = np.concatenate((intensity_attribute, points[:,9:10]/255), axis=1)
    lane_candidate_attribute = np.concatenate((intensity_attribute, points[:,10:11]/255), axis=1)
    lane_attribute = np.concatenate((rgb, points[:,11:12]/255), axis=1)
    v = pptk.viewer(xyz)
    v.set(point_size=0.005)
    v.attributes(intensity_attribute, plane_attribute, lane_candidate_attribute, lane_attribute, rgb)
    v.set(bg_color = [0,0,0,1])
    v.set(show_grid = False)
    v.set(show_info = False)
    v.set(show_axis = False)

if __name__ == '__main__':
    run()
