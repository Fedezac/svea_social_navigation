#! /usr/bin/env python3
# Plain python imports
import numpy as np
import re
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

class SocialMeasurement(object):
    SVEA_FILE = '/home/federico/universita/thesis_ws/ws/src/svea_social_navigation/data/svea_states.txt'
    PEDESTRIAN_FILE = '/home/federico/universita/thesis_ws/ws/src/svea_social_navigation/data/pedestrian_states.txt'
    # Proxemic zones radii in meters
    INTIMATE_RADIUS = 0.4
    PERSONAL_RADIUS = 1.0
    SOCIAL_RADIUS = 4.0

    def __init__(self):
        """
        Init method for the SocialMeasurement class
        """
        # Clear both files
        # TODO: uncomment when running experiments (to erase debug file)
        #open(self.SVEA_FILE, 'w').close()
        #open(self.PEDESTRIAN_FILE, 'w').close()
        # Open files in append plus read mode
        self.svea_file = open(self.SVEA_FILE, 'a+')
        self.pedestrian_file = open(self.PEDESTRIAN_FILE, 'a+')

    def add_robot_pose(self, state, timestamp):
        """
        Method to add to the robot poses array a new pose

        :param pose: robot's pose (x, y, v, yaw)
        :type pose: list[float]
        """
        robot_state = state.copy()
        robot_state.append(timestamp)
        self.svea_file.write(str(robot_state) + '\n')

    def add_pedestrian_pose(self, pedestrian_id, state, timestamp):
        """
        Method to add to the pedestrian poses array a new pose

        :param pose: robot's pose (x, y, v, yaw)
        :type pose: list[float]
        """
        ped_state = np.append(state, timestamp)
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=np.inf)
        self.pedestrian_file.write(str({pedestrian_id: ped_state}) + '\n')

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

    def plot_traj(self):
        """
        Method to plot robot's and pedestrian's trajectory, plus proxemics spaces
        """
        fig_traj, ax_traj = plt.subplots(num='Trajectory')
        fig_traj.set_dpi(200)
        for key in self.pedestrian_states:
            for i, pose in enumerate(self.pedestrian_states[key]):
                ax_traj.clear()
                intimate_proxemic = mpatches.Circle((pose[0], pose[1]), self.INTIMATE_RADIUS, edgecolor='orange', fill=False, alpha=0.5)
                personal_proxemic = mpatches.Circle((pose[0], pose[1]), self.PERSONAL_RADIUS, edgecolor='green', fill=False, alpha=0.5)
                social_proxemic = mpatches.Circle((pose[0], pose[1]), self.SOCIAL_RADIUS, edgecolor='blue', fill=False, alpha=0.5)
                ax_traj.add_artist(intimate_proxemic)
                ax_traj.add_artist(personal_proxemic)
                ax_traj.add_artist(social_proxemic)
                ax_traj.plot(self.svea_states[i][0], self.svea_states[i][1], 'ro', markersize=2)
                ax_traj.plot(pose[0], pose[1], 'bo', markersize=2)
                ax_traj.set_xlim(-2, 15)
                ax_traj.set_ylim(-2, 15)
                ax_traj.legend(['Intimate Space', 'Personal Space', 'Social Space', 'Robot Position', 'Pedestrian Position'], fontsize='x-medium')
                plt.draw()

    def plot_psit(self):
        """
        Method used for plotting PSIT (used to measure how much time the robot spends in the personal space of a human)
        """
        for key in self.pedestrian_states:
            intimate_psit = 0
            intimate_invasion = False
            intimate_plot = []

            personal_psit = 0
            personal_invasion = False
            personal_plot = []

            social_psit = 0
            social_invasion = False
            social_plot = []
            plt.ion()
            fig_psit, ax_psit = plt.subplots(num=f'PSIT (Personal Space Invasion Time) Pedestrian {key}')
            fig_psit.set_dpi(200)
            for i, pose in enumerate(self.pedestrian_states[key]):
                if (self.svea_states[i][0] - pose[0]) ** 2 + (self.svea_states[i][1] - pose[1]) ** 2 <= self.INTIMATE_RADIUS ** 2:
                    if intimate_invasion:
                        intimate_psit += self.svea_states[i][4] - self.svea_states[i - 1][4]
                    intimate_invasion = True
                else:
                    intimate_invasion = False
                if (self.svea_states[i][0] - pose[0]) ** 2 + (self.svea_states[i][1] - pose[1]) ** 2 <= self.PERSONAL_RADIUS ** 2:
                    if personal_invasion:
                        personal_psit += self.svea_states[i][4] - self.svea_states[i - 1][4]                      
                    personal_invasion = True
                else:
                    personal_invasion = False
                if (self.svea_states[i][0] - pose[0]) ** 2 + (self.svea_states[i][1] - pose[1]) ** 2 <= self.SOCIAL_RADIUS ** 2:
                    if social_invasion:
                        social_psit += self.svea_states[i][4] - self.svea_states[i - 1][4]
                    social_invasion = True
                else:
                    social_invasion = False
                intimate_plot.append([self.svea_states[i][4] - self.svea_states[0][4], intimate_psit])
                personal_plot.append([self.svea_states[i][4] - self.svea_states[0][4], personal_psit])
                social_plot.append([self.svea_states[i][4] - self.svea_states[0][4], social_psit])
                ax_psit.plot(np.array(intimate_plot)[:, 0], np.array(intimate_plot)[:, 1], '-r', linewidth=1)
                ax_psit.plot(np.array(personal_plot)[:, 0], np.array(personal_plot)[:, 1], '-b', linewidth=1)
                ax_psit.plot(np.array(social_plot)[:, 0], np.array(social_plot)[:, 1], '-g', linewidth=1)
            ax_psit.set_xlim(-0.2, int(self.svea_states[-1][4] - self.svea_states[0][4]) + 1)
            ax_psit.set_ylim(-0.2, int(self.svea_states[-1][4] - self.svea_states[0][4]))
            ax_psit.set_xlabel('t [s]')
            ax_psit.set_ylabel('Invasion Time [s]')
            ax_psit.legend(['Intimate Space Invasion Time', 'Personal Space Invasion Time', 'Social Space Invasion Time'], fontsize='medium')

    def plot_sii_over_time(self):
        """
        Method for plotting SII over time (used to measure how close the robot is to thu human with a strong focus on the
        psychological aspect)
        """
        # Psychological threshold for human safety (is SII > Tp the human might feel uncomfortable)
        Tp = 0.54
        sigma = self.PERSONAL_RADIUS / 2
        sii_plot = []
        sii = []
        plt.ion()
        fig_sii, ax_sii = plt.subplots(num='SII (Social Individual Index) Over Time')
        fig_sii.set_dpi(200)
        for i in range(len(self.pedestrian_states[0])):
            for key in self.pedestrian_states:
                sii.append(np.exp(-(((self.svea_states[i][0] - self.pedestrian_states[key][i][0]) / (np.sqrt(2) * sigma)) ** 2  + ((self.svea_states[i][1] - self.pedestrian_states[key][i][1]) / (np.sqrt(2) * sigma)) ** 2)))
            sii_plot.append([self.svea_states[i][4] - self.svea_states[0][4], np.max(sii)])
            sii = []
            ax_sii.plot(np.array(self.svea_states)[:, 4] - self.svea_states[0][4], np.full(np.shape(self.svea_states)[0], Tp), '-r', linewidth=1)
            ax_sii.plot(np.array(sii_plot)[:, 0], np.array(sii_plot)[:, 1], '-co', markersize=2, linewidth=1)
        ax_sii.set_xlim(0, int(self.svea_states[-1][4] - self.svea_states[0][4]) + 1)
        ax_sii.set_ylim(0, 1.5)
        ax_sii.set_xlabel('t [s]')
        ax_sii.set_ylabel('SII')
        ax_sii.legend(['Tp Psychological Threshold', 'SII'], fontsize='medium')
        plt.draw()
        plt.show(block=False)

    def plot_rmi_over_time(self):
        """
        Method for plotting RMI over time (used to measure the relative motion between a robot and a human)
        """
        Tm = 2.2
        rmi_plot = []
        rmi = []
        plt.ion()
        fig_rmi, ax_rmi = plt.subplots(num='RMI (Relative Motion Index) Over Time')
        fig_rmi.set_dpi(200)
        for i in range(len(self.pedestrian_states[0])):
            for key in self.pedestrian_states:
                beta = self.svea_states[i][3] - np.arctan2(self.svea_states[i][1] - self.pedestrian_states[key][i][1], self.svea_states[i][0] - self.pedestrian_states[key][i][0])
                phi = self.pedestrian_states[key][i][3] - np.arctan2(self.pedestrian_states[key][i][1] - self.svea_states[i][1], self.pedestrian_states[key][i][0] - self.svea_states[i][1])
                rmi.append((2 + self.svea_states[i][2] * np.cos(beta) + self.pedestrian_states[key][i][2] * np.cos(phi)) / np.sqrt((self.pedestrian_states[key][i][0] - self.svea_states[i][0]) ** 2 + (self.pedestrian_states[key][i][1] - self.svea_states[i][1]) ** 2))
            rmi_plot.append([self.svea_states[i][4] - self.svea_states[0][4], np.max(rmi)])
            rmi = []
            ax_rmi.plot(np.array(self.svea_states)[:, 4] - self.svea_states[0][4], np.full(np.shape(self.svea_states)[0], Tm), '-r', linewidth=1)
            ax_rmi.plot(np.array(rmi_plot)[:, 0], np.array(rmi_plot)[:, 1], '-co', markersize=2, linewidth=1)
        ax_rmi.set_xlim(0, int(self.svea_states[-1][4] - self.svea_states[0][4]) + 1)
        ax_rmi.set_ylim(0, 7)
        ax_rmi.set_xlabel('t [s]')
        ax_rmi.set_ylabel('RMI')
        ax_rmi.legend(['Tm Psychological Threshold', 'RMI'], fontsize='medium')
        plt.draw()
        plt.show(block=False)

    def plot_sii(self):
        """
        Method for plotting SII over path waypoints (used to measure how close the robot is to thu human with a strong focus on the
        psychological aspect)
        """
        # Psychological threshold for human safety (is SII > Tp the human might feel uncomfortable)
        Tp = 0.54
        sigma = self.PERSONAL_RADIUS / 2
        sii = []
        sii_plot = []
        plt.ion()
        fig_sii, ax_sii = plt.subplots(num='SII (Social Individual Index)')
        fig_sii.set_dpi(200)
        colors = 'bgrcmyk'
        for i in range(len(self.pedestrian_states[0])):
            for key in self.pedestrian_states:
                sii.append(np.exp(-(((self.svea_states[i][0] - self.pedestrian_states[key][i][0]) / (np.sqrt(2) * sigma)) ** 2  + ((self.svea_states[i][1] - self.pedestrian_states[key][i][1]) / (np.sqrt(2) * sigma)) ** 2)))
            sii_plot.append([i, np.max(sii)])
            sii = []
            ax_sii.plot([*range(len(self.pedestrian_states[key]))], np.full(np.shape(self.svea_states)[0], Tp), '-r', linewidth=1)
            ax_sii.plot(np.array(sii_plot)[:, 0], np.array(sii_plot)[:, 1], '-co', markersize=2, linewidth=1)
        ax_sii.set_xlim(0, len(self.pedestrian_states[key]))
        ax_sii.set_ylim(0, 1.5)
        ax_sii.set_xlabel('Path Point')
        ax_sii.set_ylabel('SII')
        ax_sii.legend(['Tp Psychological Threshold', 'SII'], fontsize='medium')
        plt.draw()
        plt.show(block=False)

    def plot_rmi(self):
        """
        Method for plotting RMI over path waypoints (used to measure the relative motion between a robot and a human)
        """
        Tm = 2.2
        rmi_plot = []
        rmi = []
        plt.ion()
        fig_rmi, ax_rmi = plt.subplots(num='RMI (Relative Motion Index)')
        fig_rmi.set_dpi(200)
        colors = 'bgrcmyk'
        for i in range(len(self.pedestrian_states[0])):
            for key in self.pedestrian_states:
                beta = self.svea_states[i][3] - np.arctan2(self.svea_states[i][1] - self.pedestrian_states[key][i][1], self.svea_states[i][0] - self.pedestrian_states[key][i][0])
                phi = self.pedestrian_states[key][i][3] - np.arctan2(self.pedestrian_states[key][i][1] - self.svea_states[i][1], self.pedestrian_states[key][i][0] - self.svea_states[i][1])
                rmi.append((2 + self.svea_states[i][2] * np.cos(beta) + self.pedestrian_states[key][i][2] * np.cos(phi)) / np.sqrt((self.pedestrian_states[key][i][0] - self.svea_states[i][0]) ** 2 + (self.pedestrian_states[key][i][1] - self.svea_states[i][1]) ** 2))
            rmi_plot.append([i, np.max(rmi)])
            rmi = []
            ax_rmi.plot([*range(len(self.pedestrian_states[key]))], np.full(np.shape(self.svea_states)[0], Tm), '-r', linewidth=1)
            ax_rmi.plot(np.array(rmi_plot)[:, 0], np.array(rmi_plot)[:, 1], '-co', markersize=2, linewidth=1)
        ax_rmi.set_xlim(0, len(self.pedestrian_states[key]))
        ax_rmi.set_ylim(0, 7)
        ax_rmi.set_xlabel('Path Point')
        ax_rmi.set_ylabel('RMI')
        ax_rmi.legend(['Tm Psychological Threshold', 'RMI'], fontsize='medium')


if __name__ == '__main__':
    m = SocialMeasurement()
    m.read_robot_poses()
    m.read_pedestrian_poses()
    #m.plot_traj()
    m.plot_psit()
    m.plot_sii_over_time()
    m.plot_rmi_over_time()
    m.plot_sii()
    m.plot_rmi()
    input("Press Enter to continue...")