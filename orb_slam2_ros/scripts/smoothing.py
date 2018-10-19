#!/usr/bin/python

import numpy as np
from scipy import interpolate
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, Pose
import math
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import networkx as nx

waypoints = None
last_num_points = None  # Last Time stamp
left_line = None
right_line = None
latest_midline = False


# Gets called whenever we receive new waypoints
# Process them step by step and publishes the results
def callback(msg):
    global waypoints

    waypoints = readPoints(msg)
    num_points = waypoints.shape[0]

    waypoints = findLoop(waypoints, radius=0.5)

    waypoints = filterPoints(waypoints, 0.3, 5.0)

    waypoints = smoothLine(waypoints, num_points)

    waypoints_pub.publish(get_waypoint_marker(generatePoints(waypoints), "map", 0, 1, 0))


# Inputs points and frame
# convert those to a visualizsation messages marker msg
def get_waypoint_marker(points, frame, r=1, g=0, b=0):
    m1 = Marker()
    m1.header.frame_id = frame
    m1.header.stamp = rospy.Time.now()
    m1.type = Marker.POINTS
    m1.points = points
    m1.action = Marker.ADD
    m1.color.a = 1.0
    m1.color.r = r
    m1.color.g = g
    m1.color.b = b
    m1.scale.x = 0.5
    m1.scale.y = 0.5
    m1.scale.z = 0.5
    m1.pose = Pose()

    return m1


# Find a loop from the list of frames
def findLoop(points, radius=0.5):
    # frames comes in a order list, so we can just go through them sequentially
    new_points = points[:2]
    start = points[0]  # always [0, 0] with ORBSLAM
    left_start = False

    for each in points[2:]:
        dist = math.hypot(each[0] - start[0], each[1] - start[1])
        if dist > radius:
            new_points = np.append(new_points, np.reshape(each, (1, 2)), axis=0)
            if not left_start:
                left_start = True
                rospy.loginfo("Successfully left the start")

        else:
            if not left_start:
                new_points = np.append(new_points, np.reshape(each, (1, 2)), axis=0)
            else:
                rospy.loginfo("Successfully found the track loop")
                break

    if points.shape == new_points.shape:
        rospy.logwarn("Wasn't able to close the loop")

    return new_points


# Smooth the line ana generated nice and uniform waypoints
def smoothLine(centerline, out_pts=None):
    """Centerline is 2xN (x y coordinates of midpoints) np array
        close_track (Connect first and last point)
        track limit track widtd
    """

    if out_pts is None:
        out_pts = 1000

    tck, u = interpolate.splprep(centerline.T, s=4.0, per=True)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, out_pts), tck)
    return np.array((x_i, y_i)).T


# reads a Marker msg and converts it to a numpy array
# we do not care about the z axis
def readPoints(msg):
    xs = []
    ys = []
    for point in msg.points:
        xs.append(point.x)
        ys.append(point.y)

    # create array of point locations and return it
    return np.column_stack((xs, ys))


# merges close points so that we have a purer dataset
# removes old waypoints which might interfere with the new ones
def filterPoints(points, rad=0.3, remap_thresh=2.0):
    global last_num_points
    to_drop = []

    # Now remove all old irrelevant frames
    if (last_num_points is not None):
        old_points = points[:last_num_points]
        old_points_idx = np.argwhere(points[:last_num_points])
        print("Found old points", old_points.shape[0])

        new_points_idx = np.argwhere(points[last_num_points:])
        print("Found new points", new_points_idx.shape[0])

        for each in new_points_idx:
            new_pts = np.repeat(np.reshape(points[each[0]], (1, 2)), np.size(old_points, axis=0), axis=0)
            dist = np.linalg.norm(new_pts - old_points, axis=1)
            points_to_remove = np.argwhere(dist < remap_thresh)

            for point in points_to_remove:
                to_drop.append(int(old_points_idx[point[0], 0]))

    # remove all old points
    # for each in old_points_idx:
        # to_drop.append(each[0])

    to_drop_set = np.array(to_drop)
    to_drop_set = np.unique(to_drop_set)
    print("deleting points")
    print(to_drop_set.shape[0])

    points = np.delete(points, to_drop_set, axis=0)
    # print(type(to_drop))

    # Now fuse all points which are close together
    # pairs_to_fuse = np.arange(10)
    # while pairs_to_fuse.size != 0:
    to_drop = []
    tree = cKDTree(points)

    # find pairs which are close to each other
    pairs_to_fuse = tree.query_pairs(r=rad, output_type='ndarray')
    print("Found", len(pairs_to_fuse), "pairs")

    # now replace the first of the pair with their average
    for each in pairs_to_fuse:
        points[each[0]] = np.average(points[each], axis=0)
        # points[each[0], 2] = max(points[each[0], 2], points[each[1], 2])
        to_drop.append(each[1])

    last_num_points = points.shape[0]

    to_drop_set = np.array(to_drop)
    to_drop_set = np.unique(to_drop_set)
    print("deleting points")
    print(to_drop_set.shape[0])

    points = np.delete(points, to_drop_set, axis=0)

    return points


# Use a unsupervised Nearest naighbour search to find the correct
# order of points around the track
# Needed before the smoothing stage
def orderPoints(points, radius=2.0, n_neighbors=4):
    clf = NearestNeighbors(2, radius=radius).fit(points)
    G = clf.kneighbors_graph(n_neighbors=n_neighbors)

    T = nx.from_scipy_sparse_matrix(G)

    order = list(nx.dfs_preorder_nodes(T, 0))

    xx = points[order, 0]
    yy = points[order, 1]

    points = np.array(list(zip(xx, yy)))

    return points


# Converts a numpy array into an array of Points()
def generatePoints(waypoints):
    points = []
    for val in waypoints:
        point = Point()
        point.x = val[0]
        point.y = val[1]
        points.append(point)

    # print(points)
    return points


if __name__ == '__main__':

    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('trajectory_smoother', anonymous=True)

    rospy.Subscriber("/orb_slam2_ros/waypoints", Marker, callback)
    waypoints_pub = rospy.Publisher("/orb_slam2_ros/smooth_waypoints", Marker, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
