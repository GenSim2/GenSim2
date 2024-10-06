import numpy as np


class Link:
    def __init__(self, link_name):
        self.link_name = link_name
        # Naming rule: concaten tag name as the variable name, and attribute name as the key
        self.visuals = []

    def hasVisual(self):
        if len(self.visuals) == 0:
            return False
        return True

    def addVisual(self, visual_name=None):
        self.visuals.append(Visual(visual_name))

    def setVisualOriginXyz(self, xyz):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].origin["xyz"] = np.array(xyz)

    def setVisualOriginRpy(self, rpy):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].origin["rpy"] = np.array(rpy)

    def setVisualGeometryMeshFilename(self, filename):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].geometry_mesh["filename"] = filename

    def __repr__(self):
        output = {}
        output["name"] = self.link_name
        output["visual"] = self.visuals
        return str(output)


class Visual:
    def __init__(self, visual_name=None):
        self.visual_name = visual_name
        self.origin = {"xyz": np.array([0, 0, 0]), "rpy": np.array([0, 0, 0])}
        self.geometry_mesh = {"filename": None}

    def __repr__(self):
        output = {}
        output["origin"] = self.origin
        output["mesh"] = self.geometry_mesh["filename"]
        return str(output)
