import xml.etree.ElementTree as ET
import numpy as np
import os
from .Link import Link
from .Joint import Joint


def parseThreeNumber(string):
    strings = string.split(" ")
    numbers = np.array(list(map(float, strings)))
    return numbers


class URDFParser:
    def __init__(self, file_name):
        self.file_name = file_name
        self._root_path = os.path.dirname(file_name) + "/"
        self.links = {}
        self.joints = {}

    # Parse the URDF(XML) file into a tree structure
    def parse(self):
        # Get the XML tree
        root_xml = ET.parse(self.file_name).getroot()
        self.links_xml = root_xml.findall("link")
        self.joints_xml = root_xml.findall("joint")
        # Parse links before parsing joints
        self.parseLinks()
        self.parseJoints()

    def parseLinks(self):
        for link_xml in self.links_xml:
            link_name = link_xml.attrib["name"]
            link = Link(link_name)
            # Deal with multiple visuals
            visuals_xml = link_xml.findall("visual")
            for visual_xml in visuals_xml:
                # Add new visual in link
                if "name" in visual_xml.attrib:
                    visual_name = visual_xml.attrib["name"]
                else:
                    visual_name = None
                link.addVisual(visual_name)
                # Get origin
                origin_xml = visual_xml.find("origin")
                if origin_xml != None:
                    if "xyz" in origin_xml.attrib:
                        xyz = parseThreeNumber(origin_xml.attrib["xyz"])
                        link.setVisualOriginXyz(xyz)
                    if "rpy" in origin_xml.attrib:
                        rpy = parseThreeNumber(origin_xml.attrib["rpy"])
                        link.setVisualOriginRpy(rpy)
                # Get geometry
                geometry_xml = visual_xml.find("geometry")
                if geometry_xml != None:
                    mesh_xml = geometry_xml.find("mesh")
                    if mesh_xml != None:
                        filename = mesh_xml.attrib["filename"]
                        link.setVisualGeometryMeshFilename(self._root_path + filename)
            self.links[link_name] = link

    def parseJoints(self):
        for joint_xml in self.joints_xml:
            joint_name = joint_xml.attrib["name"]
            joint_type = joint_xml.attrib["type"]
            child_name = joint_xml.find("child").attrib["link"]
            parent_name = joint_xml.find("parent").attrib["link"]
            joint = Joint(joint_name, joint_type, child_name, parent_name)
            # Get origin
            origin_xml = joint_xml.find("origin")
            if origin_xml != None:
                if "xyz" in origin_xml.attrib:
                    xyz = parseThreeNumber(origin_xml.attrib["xyz"])
                    joint.setOriginXyz(xyz)
                if "rpy" in origin_xml.attrib:
                    rpy = parseThreeNumber(origin_xml.attrib["rpy"])
                    joint.setOriginRpy(rpy)
            # Get Axis
            axis_xml = joint_xml.find("axis")
            if axis_xml != None:
                axis = parseThreeNumber(axis_xml.attrib["xyz"])
                joint.setAxis(axis)
            # Get Limit
            limit_xml = joint_xml.find("limit")
            if limit_xml != None:
                lower = float(limit_xml.attrib["lower"])
                upper = float(limit_xml.attrib["upper"])
                joint.setLimitLower(lower)
                joint.setLimitUpper(upper)
            self.joints[joint_name] = joint
