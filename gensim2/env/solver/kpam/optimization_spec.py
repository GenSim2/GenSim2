from typing import List, Dict
from collections import OrderedDict
import copy
import yaml
import os
import IPython

import gensim2.env.solver.kpam.term_spec as term_spec
from gensim2.env.solver.kpam.term_spec import (
    OptimizationTermSpec,
    Point2PointConstraintSpec,
    Point2PointCostL2Spec,
)


class OptimizationProblemSpecification(object):
    """
    The class serves as the interface between the
    config file and solver
    """

    def __init__(
        self,
        task_name="",  # type: str
        category_name="",  # type: str
        tool_keypoint_name_list=None,  # type: List[str]
        object_keypoint_name_list=None,  # type: List[str]
    ):
        """
        The optimization spec can be constructed in python code
        or load from yaml config path. In latter case, these
        parameters can left default and use load_from_config method
        :param task_name:
        :param category_name:
        :param keypoint_name_list:
        """
        self._task_name = task_name  # type: str
        self._category_name = category_name  # type: str

        # The default construction of list
        if tool_keypoint_name_list is not None:
            self._tool_keypoint_name_list = tool_keypoint_name_list  # type: List[str]
        else:
            self._tool_keypoint_name_list = []

        if object_keypoint_name_list is not None:
            self._object_keypoint_name_list = (
                object_keypoint_name_list
            )  # type: List[str]
        else:
            self._object_keypoint_name_list = []

        # By default, nothing here
        self._cost_list = []  # type: List[OptimizationTermSpec]
        self._constraint_list = []  # type: List[OptimizationTermSpec]

        # # Build from keypoint name list
        # self._keypoint_name2idx = OrderedDict()  # type: Dict[str, int]
        # self.setup_keypoint_mapping()

        # The container for explicit nominal position
        # The nominal position can either explicit declared
        # Or implicit added using Point2PointCost/Constraint
        # self._keypoint_nominal_target_position = OrderedDict()  # type: Dict[str, List[float]]

    # def setup_keypoint_mapping(self):
    #     self._keypoint_name2idx.clear()
    #     for i in range(len(self._keypoint_name_list)):
    #         name = self._keypoint_name_list[i]
    #         self._keypoint_name2idx[name] = i

    # The access interface
    @property
    def task_name(self):
        return self._task_name

    @property
    def category_name(self):
        return self._category_name

    # @property
    # def keypoint_name2idx(self):
    #     return self._keypoint_name2idx

    @property
    def cost_list(self):
        return self._cost_list

    @property
    def constraint_list(self):
        return self._constraint_list

    # # The method to manipulate nominal target
    # def add_nominal_target_position(self, keypoint_name, nominal_target):  # type: (str, List[float]) -> bool
    #     # Check the existence of keypoint
    #     if keypoint_name not in self._keypoint_name2idx:
    #         return False

    #     # Check the shape of target
    #     if len(nominal_target) != 3:
    #         return False

    #     # OK
    #     self._keypoint_nominal_target_position[keypoint_name] = nominal_target
    #     return True

    # def get_nominal_target_position(self, keypoint_name):  # type: (str) -> (bool, List[float])
    #     # If explicitly defined
    #     if keypoint_name in self._keypoint_nominal_target_position:
    #         return True, self._keypoint_nominal_target_position[keypoint_name]

    #     # Else search for constraints
    #     for cost_term in self.cost_list:
    #         if isinstance(cost_term, Point2PointCostL2Spec):
    #             if cost_term.keypoint_name == keypoint_name:
    #                 return True, cost_term.target_position
    #     for constraint_term in self.constraint_list:
    #         if isinstance(constraint_term, Point2PointConstraintSpec):
    #             if constraint_term.keypoint_name == keypoint_name:
    #                 return True, constraint_term.target_position

    #     # Not available
    #     return False, []

    # The method to modify the specification from python
    def add_cost(self, cost_term):  # type: (OptimizationTermSpec) -> bool
        if not cost_term.is_cost():
            return False
        copied = copy.deepcopy(cost_term)
        self._cost_list.append(copied)
        return True

    def add_constraint(self, constraint_term):  # type: (OptimizationTermSpec) -> bool
        if constraint_term.is_cost():
            return False
        copied = copy.deepcopy(constraint_term)
        self._constraint_list.append(copied)
        return True

    def add_optimization_term(
        self, optimization_term
    ):  # type: (OptimizationTermSpec) -> bool
        if optimization_term.is_cost():
            return self.add_cost(optimization_term)
        else:
            return self.add_constraint(optimization_term)

    # The interface from/to yaml
    def write_to_yaml(self, yaml_save_path):  # type: (str) -> None
        data_map = dict()
        data_map["task_name"] = self._task_name
        data_map["category_name"] = self._category_name
        data_map["tool_keypoint_name_list"] = self._tool_keypoint_name_list
        data_map["object_keypoint_name_list"] = self._object_keypoint_name_list

        # # For cost terms
        # cost_map_list = []
        # for cost in self._cost_list:
        #     cost_i_map = cost.to_dict()
        #     cost_i_map["type"] = cost.type_name()
        #     cost_map_list.append(cost_i_map)
        # data_map["cost_list"] = cost_map_list

        # For constraint terms
        constraint_map_list = []
        for constraint in self._constraint_list:
            constraint_i_map = constraint.to_dict()
            constraint_i_map["type"] = constraint.type_name()
            constraint_map_list.append(constraint_i_map)
        data_map["constraint_list"] = constraint_map_list

        # Save to yaml
        with open(yaml_save_path, mode="w") as save_file:
            yaml.dump(data_map, save_file)
            save_file.close()

    def load_from_config(self, data_map):  # type: (str) -> bool
        # Basic meta
        self._task_name = data_map["task_name"]
        self._category_name = data_map["category_name"]
        self._tool_keypoint_name_list = data_map["tool_keypoint_name_list"]
        self._object_keypoint_name_list = data_map["object_keypoint_name_list"]
        # self._keypoint_name2idx.clear()
        # self.setup_keypoint_mapping()

        # For cost terms

        # cost_map_list = data_map["cost_list"]
        # self._cost_list = []
        # for cost in cost_map_list:

        #     cost_type = cost["type"]  # type: str
        #     if cost_type == term_spec.Point2PointCostL2Spec.type_name():
        #         cost_spec = term_spec.Point2PointCostL2Spec()
        #         cost_spec.from_dict(cost)
        #         self._cost_list.append(cost_spec)
        #     elif cost_type == term_spec.Point2PlaneCostSpec.type_name():
        #         cost_spec = term_spec.Point2PlaneCostSpec()
        #         cost_spec.from_dict(cost)
        #         self._cost_list.append(cost_spec)
        #     else:
        #         raise RuntimeError("Unknown cost type %s" % cost_type)

        # For constraint terms
        constraint_map_list = data_map["constraint_list"]
        self._constraint_list = []
        for constraint in constraint_map_list:
            constraint_type = constraint["type"]
            if constraint_type == term_spec.Point2PointConstraintSpec.type_name():
                constraint_spec = term_spec.Point2PointConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif (
                constraint_type
                == term_spec.KeypointAxisParallelConstraintSpec.type_name()
            ):
                constraint_spec = term_spec.KeypointAxisParallelConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif (
                constraint_type
                == term_spec.KeypointAxisOrthogonalConstraintSpec.type_name()
            ):
                constraint_spec = term_spec.KeypointAxisOrthogonalConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif (
                constraint_type == term_spec.FrameAxisParallelConstraintSpec.type_name()
            ):
                constraint_spec = term_spec.FrameAxisParallelConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif (
                constraint_type
                == term_spec.FrameAxisOrthogonalConstraintSpec.type_name()
            ):
                constraint_spec = term_spec.FrameAxisOrthogonalConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            else:
                raise RuntimeError("Unknown constraint type %s" % constraint_type)
        return True
