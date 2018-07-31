#!/usr/bin/env python2
import rospy
from std_msgs.msg import Int8MultiArray, Int16, Bool
from geometry_msgs.msg import PoseStamped, PoseArray
import numpy as np
import matplotlib.pyplot as plt

def targetReached(val):
    global planner_target
    planner_target = None
def plannerUpdate(msg):
    print("received new planner target/trajectory")
    global planner_target
    planner_target = msg
def pathUpdate(int8array):
    print("received path")
    global path
    data = np.array(int8array.data)
    path = np.reshape(data, (int8array.layout.dim[0].size, int8array.layout.dim[1].size))
def mapUpdate(int8array):
    print("received new map")
    global env_map
    data = np.array(int8array.data)
    env_map = np.reshape(data, (int8array.layout.dim[0].size, int8array.layout.dim[1].size))

def checkPath(event):
    if planner_target is None or path is None:
        return
    for x,y in path:
        if env_map[y, x] == 0:
            global path
            print(y, x)
            #plt.matshow(env_map)
            #plt.show()
            print("path became blocked. Recalculating...")
            abortPub.publish(Bool(True))
            if isinstance(planner_target, PoseStamped):
                replanPub_target.publish(planner_target)
            elif isinstance(planner_target,PoseArray):
                replanPub_trajectory.publish(planner_target)
            else:
                print("cannot replan, unknown target type")
            path = None
            return

planner_target = None
path = None
rospy.init_node("path_monitor")
abortPub = rospy.Publisher("/direct_move/abort", Bool, queue_size=2)
replanPub_target = rospy.Publisher("/local_planner/target", PoseStamped, queue_size=1)
replanPub_trajectory = rospy.Publisher("/local_planner/trajectory", PoseArray, queue_size=1)
rospy.Timer(rospy.Duration(1), checkPath)
rospy.Subscriber("/local_planner/target", PoseStamped, plannerUpdate)
rospy.Subscriber("/local_planner/trajectory", PoseArray, plannerUpdate)
rospy.Subscriber("/direct_move/reached_target", Bool, targetReached)
rospy.Subscriber("/local_planner/path", Int8MultiArray, pathUpdate)
rospy.Subscriber("/local_planner/map", Int8MultiArray, mapUpdate, queue_size=1)
rospy.spin()
