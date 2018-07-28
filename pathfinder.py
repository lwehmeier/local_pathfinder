#!/usr/bin/python
import math
from math import sin, cos, pi, radians, sqrt
import rospy
import tf
from std_msgs.msg import Float32, Int32, Int16, Int8
from geometry_msgs.msg import Vector3, Quaternion, Pose, PoseStamped, PoseArray
import struct
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.core.heuristic import manhatten, euclidean, octile, null
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.bi_a_star import BiAStarFinder
from pathfinding.finder.ida_star import IDAStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder
import numpy as np
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import matplotlib.pyplot as plt
from local_pathfinder.msg import Target
plt.style.use('classic')
BASE_POTENTIAL = 50.0
ALLOW_UNKNOWN = True
DEBUG_PLOT = True
POTENTIAL_DISTANCE = 0.4
class MapMgr:
    def __init__(self):
        self.map = None
        self.grid = None
        self.matrix = None
        self.resolution = 0
        self.width = 0
        self.height = 0

    def setMap(self, og):
        self.map = og
        self.resolution = og.info.resolution
        self.width = og.info.width
        self.height = og.info.height
        #self.matrix = self.grid2matrix(og)

    def gridVal2Matrix(self, value):
        if value == 0:
            return 1.0 #map is free
        if value <= 50:
            return 0.3 #maybe free?
        if value >50 :
            return 0.0#occupied
        if value == -1:
            return 1.0 if ALLOW_UNKNOWN else 0.0#unknown 
        print(value)


    def grid2matrix(self, occupancy_grid):
        res = occupancy_grid.info.resolution
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        data = occupancy_grid.data
        matrix = []
        for x in range(0,width):
            matrix.append([])
            for y in range(0,height):
                mval = self.gridVal2Matrix(data[height*x+y])
                matrix[x].append(mval)
        return np.array(matrix)

    def potentialPropagator(self, x0,y0, x1, y1):
        d = np.linalg.norm(np.array([x0-x1,y0-y1]))
        return np.exp(1/(d*d*d*sqrt(d)))-1
    def rPotentialPropagator(self, x0,y0, x1, y1):
        d = np.linalg.norm(np.array([x0-x1,y0-y1]))
        return (np.exp(1/(d*d*d))-1)
    def propagatePotential_impl(self, matrix, x, y, propagator):
        width = len(matrix)
        height = len(matrix[0])
        d = int(round(POTENTIAL_DISTANCE / self.resolution)) # affected tile distance
        x_min = max(x-d, 0)
        x_max = min(x+d+1, width)
        y_min = max(y-d, 0)
        y_max =  min(y+d+1,height)
        potential = np.zeros((x_max-x_min, y_max-y_min))
        if False and np.product(matrix[x_min:x, y_min:y_max]) == 0.0:
            x_min_prop = x #there's another block in that direction, no need to compute potentials
        else:
            x_min_prop = x_min
        if False and np.product(matrix[x_min:x_max, y_min:y])==0.0: # block in range in y direction
            y_min_prop = y
        else:
            y_min_prop = y_min
        for i in range(x_min_prop, x_max):
            for j in range(y_min_prop, y_max):
                if not (i == x and j == y):
                    potential[i-x_min,j-y_min]=BASE_POTENTIAL*propagator(self, x,y,i,j)
        return potential
    def propagatePotential(self, matrix, x, y):
        return self.propagatePotential_impl(matrix, x, y, MapMgr.potentialPropagator)
    def propagateRPotential(self, matrix, x, y):
        return self.propagatePotential_impl(matrix, x, y, MapMgr.rPotentialPropagator)
    def computePotential(self, matrix):
        width = len(matrix)
        height=len(matrix[0])
        potential = np.zeros((width, height))
        for x in range(0, width):
            for y in range(0, height):
                if matrix[x,y]==0.0:
                    d = int(round(POTENTIAL_DISTANCE / self.resolution)) # affected tile distance
                    x_min = max(x-d, 0)
                    x_max = min(x+d+1, width)
                    y_min = max(y-d, 0)
                    y_max =  min(y+d+1,height)
                    lp = self.propagatePotential(matrix,x,y)
                    potential[x_min:x_max, y_min:y_max] += lp
        return potential
    def computeRPotential(self, matrix):
        width = len(matrix)
        height=len(matrix[0])
        potential = np.zeros((width, height))
        for x in range(0, width):
            for y in range(0, height):
                if matrix[x,y]==0.0:
                    d = int(round(POTENTIAL_DISTANCE / self.resolution)) # affected tile distance
                    x_min = max(x-d, 0)
                    x_max = min(x+d+1, width)
                    y_min = max(y-d, 0)
                    y_max =  min(y+d+1,height)
                    lp = self.propagateRPotential(matrix,x,y)
                    potential[x_min:x_max, y_min:y_max] += lp
        return potential
    def applyPotential(self, matrix, potential):
        width = len(matrix)
        height=len(matrix[0])
        potential = matrix + potential
        for x in range(0, width):
            for y in range(0, height):
                if potential[x,y]<0.0:
                    potential[x,y]=0.0
                if matrix[x,y]==0.0:
                    potential[x,y]=0.0
        return potential

    def plotPotential(self, matrix):
        m = np.zeros((self.width, self.height))
        for x in range(0, self.width):
            for y in range(0, self.height):
                m[x,y] = matrix[self.width -1 - x,y]
        plt.matshow(matrix, cmap='RdYlGn_r');
        plt.colorbar()
        #plt.show()

    def genDummy(self):
        g = OccupancyGrid()
        g.info.resolution = 0.05
        g.info.width = 256
        g.info.height = 128
        tiles = []
        for x in range(0, 512):
            for y in range(0,512):
                tiles.append(Int8(round(np.random.random(1)[0]*0.51)*100))
        g.data = tiles
        print("generated occupancy grid")
        return g
    def addPotentials(self, grid):
        grid = self.grid2matrix(grid)
        print("converted grid to np")
        potential = self.computePotential(grid)
        print("done")
        return grid+potential
    def matrix2grid(self, matrix):
        grid = Grid(matrix=matrix)
        return grid
    def calcPath(self, startx, starty, targetx, targety, grid):
        start = grid.node(startx, starty)
        end = grid.node(targetx, targety)
        finder = DijkstraFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)#, heuristic=null)
        path, runs = finder.find_path(start, end, grid)
        print('operations:', runs, 'path length:', len(path))
        #fix path offset
        np = []
        for tile in path:
            np.append([tile[0], tile[1]])
        #print(grid.grid_str(path=path, start=start, end=end)
        return np
    def plotPath(self, path):
        x = []
        y = []
        for node in path:
            x.append(node[0])# flip axis for display
            y.append(node[1])
        plt.plot(x,y, 'b')
        plt.show()
    def scaleOverlayMap(self, og):
        mres = og.info.resolution
        matrix = self.grid2matrix(og)
        if mres == self.resolution:
            return matrix #no scaling required
        scaled = np.zeros((self.width, self.height))
        sf = mres/self.resolution
        if np.linalg.norm(sf - round(sf)) >= 0.01: #no integer scaling factor, abort
            print("Error: invalid map scaling factor of:" +str(sf))
            return None
        sf = int(round(sf))
        print(sf)
        for x in range(0, og.info.width):
            for y in range(0, og.info.width):
                print(matrix[x,y])
                scaled[x*sf:x*sf+sf, y*sf:y*sf+sf] = matrix[x,y]
        return scaled
    def applyOverlay(self, matrix, overlay):
        z = np.zeros((len(matrix), len(matrix[0])))
        ret = np.array(matrix+overlay)
        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                if matrix[x, y] * overlay[x,y] == 0:
                    ret[x,y] = 0.0
        return ret
    def downscale(self, sf, matrix):
        if int(sf) != sf:
            print("ERROR: invalid downscaling factor. Only integers are supported")
            return
        sf = int(sf)
        height = self.height/sf
        width = self.width/sf
        m = np.zeros((width, height))
        for x in range(0, width):
            for y in range(0, height):
                tile = np.sum(matrix[x*sf:x*sf+sf, y*sf:y*sf+sf])
                tile = tile / (1.0*sf*sf)
                containsZero = 0.0 == np.product(matrix[x*sf:x*sf+sf, y*sf:y*sf+sf])
                #print(matrix[x*sf:x*sf+sf, y*sf:y*sf+sf])
                #print(tile)
                #print("["+str(x*sf)+":"+str(x*sf+sf)+","+str(y*sf)+":"+str(y*sf+sf)+"]")
                m[x, y] = 0.0 if containsZero else tile
        self.width = width
        self.height = height
        self.matrix = m
        self.resolution = self.resolution * sf
        return m
def overlayScaler(tile):
    return 0.0 if tile <= 1.0 else tile

def callbackCostmap(og):
    print("received new costmap")
    global cmap
    cmap = og
    mgr.setMap(og)
    #matrix = mgr.grid2matrix(og)
    #matrix = mgr.downscale(4, matrix)
    #mgr.matrix = matrix
def callbackUpdate(costmap_u):
    if processing:
        print("not updating map, it is currently in use by the planner")
        return
    if cmap is None:
        print("Node has not received complete costmap yet, cannot process costmap updates")
        return
    if costmap_u.x != 0 or costmap_u.y != 0:
        print("cannot handle costmap update")
        return
    global cmap
    cmap.data = costmap_u.data
    callbackCostmap(cmap)

def path2trajectory(path):
    trajectory = PoseArray()
    trajectory.header.frame_id = "map"
    poses = []
    scaling = mgr.resolution
    path = path[0:] #mk deep copy
    for i in range(0, len(path)):
        path[i] = [path[i][0]-12.8/scaling, path[i][1]-12.8/scaling]
    start = Pose()
    start.orientation = Quaternion(0,0,0,1.0)
    start.position = Vector3(path[0][0]*scaling, path[0][1]*scaling, 0)
    #poses.append(start)
    lastTile = path[0]
    direction = [path[1][0]-path[0][0], path[1][1]-path[0][1]]
    for tile in path[1:]:
        ndir = [tile[0]-lastTile[0], tile[1]-lastTile[1]]
        if ndir != direction:
            pose = Pose()
            pose.orientation = Quaternion(0,0,0,1)
            pose.position = Vector3(lastTile[0]*scaling, lastTile[1]*scaling, 0)
            poses.append(pose)
        lastTile=tile
        direction=ndir
    tgt = Pose()
    tgt.orientation = Quaternion(0,0,0,1)
    tgt.position = Vector3(lastTile[0]*scaling, lastTile[1]*scaling, 0)
    poses.append(tgt)
    trajectory.poses = poses
    print("computed new path:")
    print(trajectory.poses)
    return trajectory

def callbackTarget(target):
    print("received new target")
    global processing
    processing = True
    matrix = mgr.grid2matrix(mgr.map)
    matrix = mgr.downscale(4, matrix)
    mgr.matrix = matrix
    overlay = mgr.scaleOverlayMap(target.allowedEnv)
    print("computed restriction overlay")
    rpot = mgr.computeRPotential(overlay)
    overlay = mgr.applyPotential(overlay, rpot)
    #grid=matrix
    potentials = mgr.computePotential(matrix)
    print("computed map potentials")
    grid = mgr.applyPotential(matrix, potentials)
    grid = mgr.applyOverlay(grid, overlay)
    print("applied potentials")
    if DEBUG_PLOT:
        mgr.plotPotential(grid)
    grid = mgr.matrix2grid(grid)
    scaling = mgr.resolution
    tgt = [(target.target.position.x + target.allowedEnv.info.origin.position.x)/scaling, (target.target.position.y + target.allowedEnv.info.origin.position.y) / scaling]
    path = mgr.calcPath(int(round((robot_pose[0] - target.allowedEnv.info.origin.position.x) / scaling)),int(round((robot_pose[1] - target.allowedEnv.info.origin.position.y)/scaling)), int(round(tgt[0])), int(round(tgt[1])), grid)
    trajectory = path2trajectory(path)
    if DEBUG_PLOT:
        mgr.plotPath(path)
    trajectoryPub.publish(trajectory)
    print(trajectory)
    processing = False
def callbackPosition(pose):
    global robot_pose
    robot_pose = [pose.pose.position.x, pose.pose.position.y, 0]

global cmap
cmap = None
global processing
processing = False
global mgr
mgr = MapMgr()
rospy.init_node('local_pathfinder')
rospy.Subscriber("/slam_out_pose", PoseStamped, callbackPosition)
rospy.Subscriber("/direct_move/target", Target, callbackTarget, tcp_nodelay=True, queue_size=5)
trajectoryPub = rospy.Publisher("/direct_move/trajectory", PoseArray, queue_size=1)
rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, callbackCostmap)
rospy.Subscriber("/move_base/global_costmap/costmap_updates", OccupancyGridUpdate, callbackUpdate, queue_size=1)
rospy.spin()
