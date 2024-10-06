import os
import gensim2
from pathlib import Path
import numpy as np
import math
import random

GENSIM_DIR = Path(gensim2.__path__._path[-1])
ASSET_ROOT = GENSIM_DIR.parent / "assets"
ARTICULATED_OBJECTS_ROOT = ASSET_ROOT / "articulated_objs"
RIGIDBODY_OBJECTS_ROOT = ASSET_ROOT / "rigidbody_objs" / "models"

ALL_ARTICULATED_OBJECTS = [
    os.path.splitext(f)[0] for f in os.listdir(ARTICULATED_OBJECTS_ROOT)
]
ALL_ARTICULATED_INSTANCES = dict()
for obj in ALL_ARTICULATED_OBJECTS:
    ALL_ARTICULATED_INSTANCES[obj] = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(ARTICULATED_OBJECTS_ROOT / obj)]
    )
ALL_LONGHORIZON_ARTICULATED_INSTANCES = {
    "drawer": ["19179", "20279", "20555", "26525", "20411", "31249", "22692", "29921"],
}
ALL_RIGIDBODY_OBJECTS = [
    os.path.splitext(f)[0] for f in os.listdir(RIGIDBODY_OBJECTS_ROOT)
]

DEFAULT_ASSET_ID = {
    "box_rotate": "100221",
    "bucket_swing": "100452",
    "drawer": "19179",
    "faucet": "148",
    "laptop_rotate": "11395",
    "toaster_press": "103559",
    "coffee_machine": "103069",
}


def get_asset_id(asset_cls, random=False, task_type="articulated"):
    if isinstance(asset_cls, list):
        asset_id = []
        for cls in asset_cls:
            asset_id.append(get_asset_id(cls))
        return asset_id
    else:
        if task_type == "articulated":
            if asset_cls in ALL_ARTICULATED_OBJECTS:
                asset_root = ARTICULATED_OBJECTS_ROOT / asset_cls
                if random:
                    asset_id = np.random.choice(os.listdir(asset_root))
                else:
                    if asset_cls in DEFAULT_ASSET_ID:
                        asset_id = DEFAULT_ASSET_ID[asset_cls]
                    else:
                        asset_id = sorted(os.listdir(asset_root))[0]
                return asset_id
            else:
                return ""
        elif task_type == "longhorizon":
            if asset_cls in ALL_LONGHORIZON_ARTICULATED_INSTANCES:
                if random:
                    asset_id = np.random.choice(
                        ALL_LONGHORIZON_ARTICULATED_INSTANCES[asset_cls]
                    )
                else:
                    asset_id = ALL_LONGHORIZON_ARTICULATED_INSTANCES[asset_cls][0]
                return asset_id
            elif asset_cls in ALL_ARTICULATED_OBJECTS:
                asset_root = ARTICULATED_OBJECTS_ROOT / asset_cls
                if random:
                    asset_id = np.random.choice(os.listdir(asset_root))
                else:
                    if asset_cls in DEFAULT_ASSET_ID:
                        asset_id = DEFAULT_ASSET_ID[asset_cls]
                    else:
                        asset_id = sorted(os.listdir(asset_root))[0]
                return asset_id
            else:
                return ""


def get_train_asset_id(asset_cls, task_type="articulated"):
    if task_type == "articulated":
        if len(ALL_ARTICULATED_INSTANCES[asset_cls]) < 9:
            return get_asset_id(asset_cls, random=True, task_type="articulated")
        else:
            train_instance = len(ALL_ARTICULATED_INSTANCES[asset_cls]) - math.ceil(
                len(ALL_ARTICULATED_INSTANCES[asset_cls]) / 10
            )
            return random.choice(ALL_ARTICULATED_INSTANCES[asset_cls][-train_instance:])
    elif task_type == "longhorizon":
        return get_asset_id(asset_cls, random=True, task_type="longhorizon")


def get_test_asset_id(asset_cls, task_type="articulated"):
    if task_type == "articulated":
        if len(ALL_ARTICULATED_INSTANCES[asset_cls]) < 9:
            return get_asset_id(asset_cls, random=True, task_type="articulated")
        else:
            test_instance = math.ceil(len(ALL_ARTICULATED_INSTANCES[asset_cls]) / 10)
            return random.choice(ALL_ARTICULATED_INSTANCES[asset_cls][:test_instance])
    elif task_type == "longhorizon":
        return get_asset_id(asset_cls, random=True, task_type="longhorizon")
