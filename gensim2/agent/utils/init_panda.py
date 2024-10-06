"""
Connect to the panda robot test. 
"""

# import rospy
import os
import panda_py
from panda_py import libfranka
import logging

hostname = "172.16.0.2"
username = "admin"
password = "wanglab123"

logging.basicConfig(level=logging.INFO)


# Connect to the robot using the Panda class.
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)


# gensim2 robot init pose
joint_pose = [
    0.00000000e00,
    -3.19999993e-01,
    0.00000000e00,
    -2.61799383e00,
    0.00000000e00,
    2.23000002e00,
    7.85398185e-01,
]

panda.move_to_joint_position(joint_pose, speed_factor=0.2)
