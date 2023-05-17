#! /usr/bin/env python3
# Plain python imports
import numpy as np
import re

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


if __name__ == '__main__':
    m = SocialMeasurement()
    m.read_robot_poses()
    m.read_pedestrian_poses()
    print(len(m.svea_states))
    print(len(m.pedestrian_states.get(0)))