import numpy as np


class Joint:
    def __init__(self, joint_name, joint_type, child_name, parent_name):
        self.joint_name = joint_name
        self.joint_type = joint_type
        self.child_name = child_name
        self.parent_name = parent_name
        # Naming rule: concaten tag name as the variable name, and attribute name as the key
        # If the tag just has one attribute, ignore the dictionary
        self.origin = {"xyz": np.array([0, 0, 0]), "rpy": np.array([0, 0, 0])}
        self.axis = np.array([1, 0, 0])
        self.limit = {"lower": 0, "upper": 0}

    def setOriginXyz(self, xyz):
        self.origin["xyz"] = np.array(xyz)

    def setOriginRpy(self, rpy):
        self.origin["rpy"] = np.array(rpy)

    def setAxis(self, axis):
        self.axis = np.array(axis)

    def setLimitLower(self, lower):
        self.limit["lower"] = lower

    def setLimitUpper(self, upper):
        self.limit["upper"] = upper

    def __repr__(self):
        output = {}
        output["name"] = self.joint_name
        output["type"] = self.joint_type
        output["child_name"] = self.child_name
        output["parent_name"] = self.parent_name
        output["origin"] = self.origin
        output["axis"] = self.axis
        output["limit"] = self.limit

        return str(output)
