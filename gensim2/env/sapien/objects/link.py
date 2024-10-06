import numpy as np
import sapien.core as sapien
from gensim2.env.base.base_object import GenSimBaseObject


class Link(GenSimBaseObject):
    def __init__(self, instance):
        self.instance: sapien.Link = instance

    @property
    def pose(self):
        return self.instance.pose

    @property
    def pos(self):
        return self.pose.p

    @property
    def quat(self):
        return self.pose.q
