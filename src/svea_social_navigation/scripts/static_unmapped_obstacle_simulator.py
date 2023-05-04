#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray

def create_marker_array(size):
    return [Marker()] * size

def create_marker(x, y, id):
    m = Marker()
    m.header.frame_id = 'map'
    m.header.stamp = rospy.Time.now()
    m.ns = 'static_unmapped_obstacle_simulator'
    m.id = id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0
    m.scale.x = 0.1
    m.scale.y = 0.1
    m.scale.z = 0.1
    m.color.a = 1.0 
    m.color.r = 0.0
    m.color.g = 1.0
    m.color.b = 0.0
    return m

def publish_obstacle_msg():
    pub = rospy.Publisher('/static_unmapped_obstacles', MarkerArray, queue_size=1, latch=True)
    rospy.init_node("static_unmapped_obstacle_simulator")
    
    obstacle_msg = MarkerArray()
    obstacle_msg.markers = create_marker_array(3)

    obstacle_msg.markers[0] = create_marker(3.5, 4, 0)
    obstacle_msg.markers[1] = create_marker(5, 5, 1)
    obstacle_msg.markers[2] = create_marker(2.8, 6.5, 2)

    pub.publish(obstacle_msg)
    rospy.spin()


if __name__ == '__main__':
    try:
        publish_obstacle_msg()
    except rospy.ROSInterruptException:
        pass
