#! /usr/bin/env python3
# Plain python imports
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

class SocialMeasurement(object):
    SVEA_FILE = '/home/federico/universita/thesis_ws/ws/src/svea_social_navigation/data/svea_states.txt'
    PEDESTRIAN_FILE = '/home/federico/universita/thesis_ws/ws/src/svea_social_navigation/data/pedestrian_states.txt'

    def __init__(self):
        """
        Init metho for the SocialMeasurement class
        """
        # Clear both files
        # TODO: uncomment when running experiments (to erase debug file)
        #open(self.SVEA_FILE, 'w').close()
        #open(self.PEDESTRIAN_FILE, 'w').close()
        # Open files in append plus read mode
        self.svea_file = open(self.SVEA_FILE, 'a+')
        self.pedestrian_file = open(self.PEDESTRIAN_FILE, 'a+')

    def add_robot_pose(self, state):
        """
        Method to add to the robot poses array a new pose

        :param pose: robot's pose (x, y, v, yaw)
        :type pose: list[float]
        """
        self.svea_file.write(str(state) + '\n')

    def add_pedestrian_pose(self, pedestrian_id, state):
        """
        Method to add to the pedestrian poses array a new pose

        :param pose: robot's pose (x, y, v, yaw)
        :type pose: list[float]
        """
        self.pedestrian_file.write(str({pedestrian_id: state}) + '\n')

    def close_files(self):
        """
        Method used to close open log files
        """
        print('Closing log files')
        self.svea_file.close()
        self.pedestrian_file.close()

    def read_robot_poses(self):
        """
        Method to read robot's poses from log file
        """
        # Arrays for svea's states
        self.svea_states = []
        # Reset file cursor to first char
        self.svea_file.seek(0)
        # Read all lines from file
        lines = self.svea_file.readlines()
        for l in lines:
            # Convert line into list of doubles and append it to the svea_states array
            self.svea_states.append([eval(num) for num in re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", l)])

    def read_pedestrian_poses(self):
        """
        Method to read robot's poses from log file
        """
        pedestrian_id = set()
        # Arrays for svea's states
        self.pedestrian_states = {}
        # Reset file cursor to first char
        self.pedestrian_file.seek(0)
        # Read all lines from file
        lines = self.pedestrian_file.readlines()
        for l in lines:
            array = np.array([eval(num) for num in re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", l)])
            if not (int(array[0]) in pedestrian_id):
                pedestrian_id.add(int(array[0]))
                # Convert line into list of doubles and append it to the svea_states array
                self.pedestrian_states.update({int(array[0]): list()})
            old_states = self.pedestrian_states.get(int(array[0]))
            old_states.append(list(array[1:]))
            self.pedestrian_states.update({int(array[0]): old_states})

    def get_personal_space(self, pose):
        # Arc1
        if pose[2] < 1.06:
            a = 0.54 * pose[2] + 0.33
        elif pose[2] < 1.26:
            a = 3.0
        else:
            a = 10.41 * pose[2] - 10.10
        b = 0.14 * a
        arc1 = [pose[0], pose[1], np.abs(b * np.cos(pose[3])), np.abs(b * np.sin(pose[3])), np.pi]

        # Arc 2
        e = np.sqrt(a ** 2 + b ** 2)
        d = a - b
        r = (e * (e - d)) / (2 * a)
        og = a - r
        g = pose[0:2] + og
        alpha = np.arctan2(b, a) 
        arc2 = [g[0], g[1], r * np.cos(pose[3]), r * np.sin(pose[3]), 2 * alpha]

        # Arc 3
        R = (e * (e + d)) / (2 * b)
        of = R - b
        f = pose[0:2] + of
        beta = np.arctan2(a, b)
        arc3 = [f[0], f[1], R * np.cos(pose[3]), R * np.sin(pose[3]), 2 * beta]

        return arc1, arc2, arc3



if __name__ == '__main__':
    m = SocialMeasurement()
    m.read_robot_poses()
    m.read_pedestrian_poses()
    for key in m.pedestrian_states:
        for pose in m.pedestrian_states[key]:
            arc1, arc2, arc3 = m.get_personal_space(pose)
            print(arc1)
            plt.clf()
            plt_arc1 = mpatches.Arc((arc1[0], arc1[1]), arc1[2], arc1[3], angle=arc1[4], color="orange")
            plt.gcf().gca().add_artist(plt_arc1)
            plt.show()  
            plt.pause(0.01)