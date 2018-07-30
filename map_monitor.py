#!/usr/bin/env python2
import rospy
from std_msgs.msg import Int8MultiArray, Int16, Bool
from geometry_msgs.msg import PoseStamped, PoseArray
import numpy as np


def targetReached(val):
    global planner_target
    planner_target = None
def plannerUpdate(msg):
    global planner_target
    planner_target = msg
def pathUpdate(int8array):
    global path
    data = np.array(int8array.data)
    path = np.reshape(data, (int8array.layout.dim[0].size, int8array.layout.dim[1].size))
def mapUpdate(int8array):
    global env_map
    data = np.array(int8array.data)
    env_map = np.reshape(data, (int8array.layout.dim[0].size, int8array.layout.dim[1].size))

def checkPath(event):
    if planner_target is None:
        return
    for x,y in path:
        if env_map[int(x), int(y)] == 0:
            print("path became blocked. Recalculating...")
            abortPub.publish(Bool(True))
            if planner_target is PoseStamped:
                replanPub_target.publish(planner_target)
            elif planer_target is PoseArray:
                replanPub_trajectory.publish(planner_target)
            else:
                print("cannot replan, unknown target type")

planner_target = None
rospy.init_node("path_monitor")
abortPub = rospy.Publisher("/direct_move/abort", Bool, queue_size=2)
replanPub_target = rospy.Publisher("/local_planner/target", PoseStamped, queue_size=1)
replanPub_trajectory = rospy.Publisher("/local_planner/trajectory", PoseArray, queue_size=1)
rospy.Timer(rospy.Duration(1), checkPath)
rospy.Subscriber("/local_planner/target", PoseStamped, plannerUpdate)
rospy.Subscriber("/local_planner/trajectory", PoseArray, plannerUpdate)
rospy.Subscriber("/direct_move/reached_target", Bool, targetReached)
rospy.Subscriber("/local_planner/path", Int8MultiArray, pathUpdate)
rospy.Subscriber("/local_planner/map", Int8MultiArray, mapUpdate)
rospy.spin()