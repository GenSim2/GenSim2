from .SceneNode import SceneNode
import numpy as np


class SceneGraph:
    def __init__(self, rootLink):
        self.root = SceneNode()
        self.controll_nodes = {}
        self.constructNode(self.root, rootLink)

    # Return all the nodes to control
    def getNodes(self):
        return self.controll_nodes

    # Print all the link information with associated joints
    def printInfo(self):
        for name, node in self.controll_nodes.items():
            node.printInfo()

    # Update all motion parameters to the lastest world coordinate instead of local coordinate
    # This need to be recalled after some joint may change because the joint may change, like the lamp
    def updateMotionWorld(self):
        self.update()
        self.root.updateMotionWorld()

    def update(self):
        self.root.update()

    def getMesh(self):
        self.update()
        # The mesh has been one scene in trimesh
        mesh = self.root.getMesh()
        return mesh

    def constructNode(self, node, link):
        node.name = link.link.link_name
        self.controll_nodes[node.name] = node
        node.joint = link.joint
        if node.joint != None:
            # Construct the joint node firstly; Deal with xyz and rpy of the node
            joint_xyz = node.joint.origin["xyz"]
            joint_rpy = node.joint.origin["rpy"]
            # The rotation of RPY
            node.rotateRPYLocal(joint_rpy)
            node.translateLocal(joint_xyz)
        # Construct the mesh nodes for multiple visuals in link
        visuals = link.link.visuals
        for visual in visuals:
            visual_node = SceneNode(node)
            node.addChild(visual_node)
            visual_node.name = node.name + "_mesh:" + visual.visual_name
            if visual.geometry_mesh["filename"] == None:
                raise RuntimeError("Invalid File path")
            visual_node.addMeshFile(visual.geometry_mesh["filename"])
            # Deal with xyz and rpy of the visual node
            visual_xyz = visual.origin["xyz"]
            visual_rpy = visual.origin["rpy"]
            visual_node.rotateRPYLocal(visual_rpy)
            visual_node.translateLocal(visual_xyz)

        # Construct node for the children
        for child in link.children:
            child_node = SceneNode(node)
            node.addChild(child_node)
            self.constructNode(child_node, child)
