#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped


def publish_obstacle_msg():
    pub = rospy.Publisher('/dynamic_obstacle', PointStamped, queue_size=1)
    rospy.init_node("dynamic_obstacle_simulator")

    x = 4
    y = 2.2
    z = 0.0
    vel_x = 0.0
    vel_y = 0.0
    range_x = 6.0
    range_y = 4.0

    obstacle_msg = PointStamped()
    obstacle_msg.header.stamp = rospy.Time.now()
    obstacle_msg.header.frame_id = 'map'
    obstacle_msg.point.x = y
    obstacle_msg.point.y = x
    obstacle_msg.point.z = z

    r = rospy.Rate(10)  # 10hz
    dt = 0.1
    while not rospy.is_shutdown():
        new_y = y + vel_y * dt
        if new_y >= range_y or new_y <= 0:
            vel_y = -vel_y
        obstacle_msg.point.y = new_y
        y = new_y

        new_x = x + vel_x * dt
        if new_x >= range_x or new_x <= 0:
            vel_x = -vel_x
        obstacle_msg.point.x = new_x
        x = new_x
        pub.publish(obstacle_msg)
        r.sleep()


if __name__ == '__main__':
    try:
        publish_obstacle_msg()
    except rospy.ROSInterruptException:
        pass
