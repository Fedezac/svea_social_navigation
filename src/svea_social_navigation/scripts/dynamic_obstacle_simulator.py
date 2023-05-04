#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped
from svea_msgs.msg import PointStampedList

def publish_obstacle_msg():
    pub = rospy.Publisher('/dynamic_obstacles', PointStampedList, queue_size=1)
    rospy.init_node("dynamic_obstacle_simulator")

    x = 2.7
    y = 4
    z = 0.0
    
    vel_x = 0.05
    vel_y = 0.0

    lb_x = 2.7
    ub_x = 3.2
    

    lb_y = 2.0
    ub_y = 4.0
    

    obstacle_msg = PointStampedList()
    obstacle_msg.points = []
    obstacle_msg.points.append(PointStamped())
    obstacle_msg.points[0].header.stamp = rospy.Time.now()
    obstacle_msg.points[0].header.frame_id = 'map'
    obstacle_msg.points[0].point.x = x
    obstacle_msg.points[0].point.y = y
    obstacle_msg.points[0].point.z = z

    r = rospy.Rate(10)  # 10hz
    dt = 0.1
    while not rospy.is_shutdown():
        new_y = y + vel_y * dt
        if new_y >= ub_y or new_y <= lb_y:
            vel_y = -vel_y
        obstacle_msg.points[0].point.y = new_y
        y = new_y

        new_x = x + vel_x * dt
        if new_x >= ub_x or new_x <= lb_x:
            vel_x = -vel_x
        obstacle_msg.points[0].point.x = new_x
        x = new_x
        pub.publish(obstacle_msg)
        r.sleep()


if __name__ == '__main__':
    try:
        publish_obstacle_msg()
    except rospy.ROSInterruptException:
        pass
