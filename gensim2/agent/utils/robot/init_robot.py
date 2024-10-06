"""
Connect to the panda robot test. 
"""

import os
import panda_py
from panda_py import libfranka
import logging

# TODO change the hostname, username, and password to the correct values
hostname = ""
username = ""
password = ""

logging.basicConfig(level=logging.INFO)

# Use the desk client to connect to the web-application
# running on the control unit to unlock brakes
# and activate FCI for robot torque control

# desk = panda_py.Desk(hostname, username, password)
# desk.unlock()
# desk.activate_fci()

# Connect to the robot using the Panda class.
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

panda.move_to_start()


joint_pose = [
    7.4547783e-02,
    -1.9822954e-01,
    1.0989897e-02,
    -2.6377580e00,
    4.5893144e-02,
    2.6391320e00,
    6.3584524e-01,
]

panda.move_to_joint_position(joint_pose, speed_factor=0.1)

gripper.move(0.0, 0.1)
