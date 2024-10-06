from .ExLink import ExLink


class URDFTree:
    # Construct the URDF tree based on the parser
    def __init__(self, links, joints):
        self.links = links
        self.joints = joints
        # Init ExLinks (extended links: include joint info in the child part; parent and child info)
        self.exLinks = {}
        self.initExlinks()
        # Build the tree and find the root (If not strictly a tree, consttruct a virtual root)
        self.buildTree()
        self.root = None
        self.findRoot()

    def initExlinks(self):
        # Create extended links list
        for link_name in self.links:
            exLink = ExLink(self.links[link_name])
            self.exLinks[link_name] = exLink

    def buildTree(self):
        for joint_name in self.joints:
            joint = self.joints[joint_name]
            # Connect child and parent through parent and children in exLink
            child_name = joint.child_name
            parent_name = joint.parent_name
            child = self.exLinks[child_name]
            parent = self.exLinks[parent_name]
            child.setJoint(joint)
            child.setParent(parent)
            parent.addChild(child)

    def findRoot(self):
        roots = []
        for link_name in self.exLinks:
            link = self.exLinks[link_name]
            if link.parent == None:
                roots.append(link)
        if len(roots) == 0:
            raise RuntimeError("Invalid: No root nodes for the URDF")
        elif len(roots) == 1:
            self.root = roots[0]
        else:
            # Construct a virtual root to connect all nodes without a parent
            self.root = ExLink(None)
            for child in roots:
                self.root.addChild(child)
                child.setParent(self.root)
