import os
import sys

current_path = os.getcwd()
sys.path.append(f"{current_path}")

import icecream
from gensim2.env.sapien.constructor import add_default_scene_light
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim

import numpy as np
import argparse
from sapien.utils import Viewer
import sapien.core as sapien
import json

from PIL import Image

ENV = "CloseSuitcaseLid"
dir = "assets/articulated_objs/suitcase_rotate_new/"

if __name__ == "__main__":
    fail_ids = []
    for id in os.listdir(dir):

        keypoint_path = f"{dir}/{id}/keypoints.json"
        target_keypoint_path = f"{dir}/{id}/keypoints_final.json"

        if not os.path.exists(keypoint_path):
            print(f"keypoint file {keypoint_path} not found")
            continue

        # if os.path.exists(target_keypoint_path):
        #     print(f"keypoint file {target_keypoint_path} already exists")
        #     continue

        # import ipdb; ipdb.set_trace()
        try:
            env = create_gensim(
                task_name=ENV,
                sim_type="Sapien",
                use_gui=False,
                eval=False,
                asset_id=id,
                headless=True,
            )
        except Exception as e:
            print(f"Failed to load {id}:", e)
            fail_ids.append(id)
            continue

        env.task.env.articulator.instance.set_qpos(np.array([0.0]))

        base_pose = env.task.env.articulator.instance.get_pose()
        link_pose = env.task.env.articulator.handle_link.get_pose()
        handle2link_relative_pose = env.task.env.articulator.handle2link_relative_pose
        handle_pose = link_pose.transform(handle2link_relative_pose)
        handle2base = base_pose.inv().transform(handle_pose)
        base2handle = handle2base.inv()

        kploc2base = json.load(open(keypoint_path))["keypoints"]
        # import ipdb; ipdb.set_trace()
        kplocs = {}
        for name, kploc in kploc2base.items():
            if name == "red" or name == "yellow":
                # if name == "red":
                kplocs[name] = base2handle.transform(sapien.Pose(kploc)).p.tolist()
            elif name == "blue":
                # elif name == "yellow":
                kplocs[name] = kploc

        # replace the keypoints in the json file
        # json_file = json.load(open(keypoint_path.replace(".json", "_final.json")))
        # json_file["keypoints"] = kplocs
        data = {
            "keypoints": kplocs,
        }
        json.dump(data, open(target_keypoint_path, "w"), indent=4)
        print("Saved keypoints to ", target_keypoint_path)

    print(fail_ids)
