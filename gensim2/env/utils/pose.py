import numpy as np
import transforms3d
from transforms3d.euler import euler2quat
import os
from gensim2.paths import *

ALL_ARTICULATED_OBJECTS = [
    os.path.splitext(f)[0] for f in os.listdir(ARTICULATED_OBJECTS_ROOT)
]
ALL_RIGIDBODY_OBJECTS = [
    os.path.splitext(f)[0] for f in os.listdir(RIGIDBODY_OBJECTS_ROOT)
]

DEFAULT_POSE_DICT = {
    # Articulated objects
    "door": {
        "8877": np.array([0.1, 0.0, 0.35, 0, 0, 0, 1]),
        "9168": np.array([0.12, 0.0, 0.3, 0, 0, 0, 1]),
    },
    "toaster_press": np.array([0.1, -0.3, 0.12, 0.92388, 0, 0, -0.38268]),
    "toaster_move": np.array([0.1, 0.0, 0.12, 1, 0, 0, 0]),
    "refrigerator": {
        "11231": np.array([0.2, 0.1, 0.3, 1, 0, 0, 0]),
        "11178": np.array([0.22, 0.1, 0.4, 1, 0, 0, 0]),
    },
    "coffee_machine": np.array([0.1, 0.0, 0.12, 1, 0, 0, 0]),
    "switch": np.array([0.15, 0.0, 0.35, 1, 0, 0, 0]),
    "toilet": np.array([0.12, 0.0, 0.2, 1, 0, 0, 0]),
    "washing_machine": np.array([0.15, 0.0, 0.22, 1, 0, 0, 0]),
    "laptop_move": np.array([0, 0.0, 0.15, 1, 0, 0, 0]),
    "laptop_rotate": np.array([0.1, 0.0, 0.15, 1, 0, 0, 0]),
    "box_move": np.array([0, 0.0, 0.1, 1, 0, 0, 0]),
    "box_rotate": {
        "47645": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "48492": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100141": np.array([0.08, 0.0, 0.08, 1, 0, 0, 0]),
        "100174": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100189": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100191": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100214": np.array([0.08, 0.0, 0.15, 1, 0, 0, 0]),
        "100221": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100243": np.array([0.08, 0.0, 0.11, 1, 0, 0, 0]),
        "100247": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "100664": np.array([0.08, 0.0, 0.12, 1, 0, 0, 0]),
        "100671": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
        "102379": np.array([0.08, 0.0, 0.08, 1, 0, 0, 0]),
        "102456": np.array([0.08, 0.0, 0.1, 1, 0, 0, 0]),
    },
    "suitcase_move": np.array([0.18, 0.0, 0.08, 0, 0.7071, 0, -0.7071]),
    "suitcase_rotate": np.array([-0.05, 0.0, 0.08, 0, 0.7071, 0, -0.7071]),
    "bag_move": np.array([0, 0.0, 0.1, 1, 0, 0, 0]),
    "bag_lift": np.array([0.1, 0.0, 0.1, 0.7071, 0, 0, 0.7071]),
    "bag_swing": np.array([0, 0.0, 0.1, 1, 0, 0, 0]),
    "dishwasher": np.array([0.15, 0.0, 0.15, 1, 0, 0, 0]),
    "faucet": {
        "148": np.array([0.03, 0.0, 0.2, 1, 0, 0, 0]),
        "693": np.array([0.0, 0.0, 0.18, 1, 0, 0, 0]),
        "822": np.array([0.03, 0.0, 0.18, 1, 0, 0, 0]),
        "857": np.array([0.03, 0.0, 0.15, 1, 0, 0, 0]),
        "991": np.array([0.03, 0.0, 0.20, 1, 0, 0, 0]),
        "1011": np.array([0.03, 0.0, 0.18, 1, 0, 0, 0]),
        "1053": np.array([0.0, 0.0, 0.18, 1, 0, 0, 0]),
        "1343": np.array([0.03, 0.0, 0.20, 1, 0, 0, 0]),
        "1370": np.array([0.06, 0.0, 0.20, 1, 0, 0, 0]),
        "1466": np.array([0.04, 0.0, 0.22, 1, 0, 0, 0]),
        "1556": np.array([0.05, 0.0, 0.22, 1, 0, 0, 0]),
        "1646": np.array([0.05, 0.0, 0.22, 1, 0, 0, 0]),
        "1832": np.array([0.05, 0.0, 0.15, 1, 0, 0, 0]),
    },
    "safe_move": np.array([0, 0.2, 0.15, 1, 0, 0, 0]),
    "safe_rotate": {
        "102387": np.array([0.2, -0.2, 0.15, 1, 0, 0, 0]),
        "102384": np.array([0.2, -0.2, 0.2, 1, 0, 0, 0]),
        "102381": np.array([0.2, -0.2, 0.28, 1, 0, 0, 0]),
        "102380": np.array([0.2, -0.2, 0.25, 1, 0, 0, 0]),
        "102318": np.array([0.2, -0.2, 0.25, 1, 0, 0, 0]),
        "102311": np.array([0.2, -0.2, 0.25, 1, 0, 0, 0]),
        "102309": np.array([0.22, -0.2, 0.30, 1, 0, 0, 0]),
        "102301": np.array([0.22, -0.2, 0.25, 1, 0, 0, 0]),
        "102278": np.array([0.22, -0.2, 0.22, 1, 0, 0, 0]),
        "101623": np.array([0.22, -0.2, 0.22, 1, 0, 0, 0]),
        "101619": np.array([0.2, -0.2, 0.25, 1, 0, 0, 0]),
        "101613": np.array([0.22, -0.2, 0.2, 1, 0, 0, 0]),
        "101599": np.array([0.22, -0.2, 0.2, 1, 0, 0, 0]),
        "101594": np.array([0.22, -0.2, 0.2, 1, 0, 0, 0]),
        "101593": np.array([0.22, -0.2, 0.22, 1, 0, 0, 0]),
        "101584": np.array([0.2, -0.2, 0.25, 1, 0, 0, 0]),
        "101564": np.array([0.22, -0.2, 0.22, 1, 0, 0, 0]),
    },
    "microwave": {
        "7119": np.array([0.2, 0.3, 0.12, 0.92388, 0, 0, 0.38268]),
        "7128": np.array([0.2, 0.1, 0.1, 1, 0, 0, 0]),
        "7167": np.array([0.22, 0.1, 0.13, 1, 0, 0, 0]),
        "7236": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
        "7263": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
        "7265": np.array([0.22, 0.1, 0.12, 1, 0, 0, 0]),
        "7296": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
        "7304": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
        "7310": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
        "7320": np.array([0.22, 0.1, 0.15, 1, 0, 0, 0]),
    },
    "window_push": {
        "102801": np.array([0.12, 0.0, 0.29, 1, 0, 0, 0]),
        "102906": np.array([0.12, 0.0, 0.26, 0, 0, 0, 1]),
        "103149": np.array([0.12, 0.0, 0.26, 0, 0, 0, 1]),
    },
    "window_rotate": {
        "103135": np.array([0.12, 0.0, 0.26, 1, 0, 0, 0]),
        "103056": np.array([0.12, 0.0, 0.26, 0, 0, 0, 1]),
    },
    "bucket_move": {
        "100464": np.array([0, 0.0, 0.07, 1, 0, 0, 0]),
        "100452": np.array([0, 0.0, 0.1, 1, 0, 0, 0]),
    },
    "bucket_lift": {
        "100464": np.array([0, 0.0, 0.07, 0.7071, 0, 0, -0.7071]),
        "100452": np.array([0, 0.0, 0.1, 0.7071, 0, 0, -0.7071]),
        "100443": np.array([0, 0.0, 0.1, 0.7071, 0, 0, -0.7071]),
        "100448": np.array([0, 0.0, 0.1, 0.7071, 0, 0, -0.7071]),
        "100454": np.array([0, 0.0, 0.1, 0.7071, 0, 0, -0.7071]),
    },
    "bucket_swing": {
        "100464": np.array([-0.05, 0.0, 0.07, 0, 0, 0, 1]),
        "100452": np.array([-0.05, 0.0, 0.1, 1, 0, 0, 0]),
        "100431": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100432": np.array([-0.05, 0.0, 0.12, 0, 0, 0, 1]),
        "100435": np.array([0, 0.0, 0.1, 0, 0, 0, 1]),
        "100438": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100439": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100441": np.array([-0, 0.0, 0.12, 1, 0, 0, 0]),
        "100443": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100444": np.array([-0, 0.0, 0.1, 0, 0, 0, 1]),
        "100448": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100454": np.array([-0.05, 0.0, 0.1, 0, 0, 0, 1]),
        "100461": np.array([-0, 0.0, 0.1, 0, 0, 0, 1]),
        "100462": np.array([-0, 0.0, 0.12, 0, 0, 0, 1]),
        "100465": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "100468": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "100472": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "100473": np.array([-0.0, 0.0, 0.1, 0, 0, 0, 1]),
        "100477": np.array([-0.0, 0.0, 0.1, 1, 0, 0, 0]),
        "100482": np.array([-0.0, 0.0, 0.1, 0, 0, 0, 1]),
        "100486": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "102352": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "102358": np.array([-0.0, 0.0, 0.12, 0, 0, 0, 1]),
        "102359": np.array([-0.0, 0.0, 0.14, 0, 0, 0, 1]),
        "102365": np.array([-0.0, 0.0, 0.1, 0, 0, 0, 1]),
        "102367": np.array([-0.0, 0.0, 0.1, 0, 0, 0, 1]),
        "102369": np.array([-0.0, 0.0, 0.1, 0, 0, 0, 1]),
    },
    "drawer": {
        "19179": np.array([0.16, 0.0, 0.13, 1, 0, 0, 0]),
        "20279": np.array([0.16, 0.1, 0.23, 1, 0, 0, 0]),
        "20411": np.array([0.16, 0, 0.25, 1, 0, 0, 0]),
        "20453": np.array([0.16, 0.1, 0.2, 1, 0, 0, 0]),
        "20555": np.array([0.16, 0, 0.22, 1, 0, 0, 0]),
        "22241": np.array([-0.05, 0, 0.18, 1, 0, 0, 0]),
        "22301": np.array([-0.05, 0, 0.28, 1, 0, 0, 0]),
        "22692": np.array([0.2, 0.0, 0.12, 1, 0, 0, 0]),
        "23472": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "24644": np.array([0.16, 0.0, 0.23, 1, 0, 0, 0]),
        "25308": np.array([0.16, 0.1, 0.2, 1, 0, 0, 0]),
        "26525": np.array([0.16, 0.0, 0.25, 1, 0, 0, 0]),
        "26670": np.array([0.16, 0.0, 0.22, 1, 0, 0, 0]),
        "27044": np.array([0.16, 0.0, 0.22, 1, 0, 0, 0]),
        "27189": np.array([0.18, 0.0, 0.2, 1, 0, 0, 0]),
        "29557": np.array([0.18, 0.0, 0.2, 1, 0, 0, 0]),
        "29921": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "31249": np.array([0.16, 0.0, 0.22, 1, 0, 0, 0]),
        "31601": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "32354": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "32761": np.array([0.16, 0.0, 0.18, 1, 0, 0, 0]),
    },
    "oven": {
        "7138": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "7179": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "7201": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
        "7221": np.array([0.16, 0.0, 0.2, 1, 0, 0, 0]),
    },
    # Rigid objects
    "mug": np.array([-0.2, -0.45, 0.03, 0.7071, 0, 0, -0.7071]),
    # box
    # "cracker_box": np.array([-0.25, -0.48, 0.05, 1, 0, 0, 0]),
    # "pudding_box": np.array([-0.2, -0.45, 0.055, 1, 0, 0, 0]),
    # # fruit
    # "apple": np.array([-0.2, -0.45, 0.06, 1, 0, 0, 0]),
    # "pear": np.array([-0.2, -0.45, 0.06, 1, 0, 0, 0]),
    # "lemon": np.array([-0.2, -0.45, 0.06, 1, 0, 0, 0]),
    # # ball
    # "baseball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "golf_ball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "soccer_ball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "softball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "tennis_ball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "racquetball": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "bowl": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "pitcher": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "bleach_cleanser": np.array([-0.2, -0.45, 0.1, 1, 0, 0, 0]),
    # "windex_bottle": np.array([-0.2, -0.45, 0.1, 1, 0, 0, 0]),
    # "sponge": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "wood_block": np.array([-0.2, -0.45, 0.08, 1, 0, 0, 0]),
    # "foam_brick": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "dice": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "cup": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "screwdriver": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "clamp": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "marker": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "fork": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    "spoon": np.array([-0.2, 0, 0.03, 1, 0, 0, 0]),
    # "knife": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "spatula": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "hammer": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    # "power_drill": np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0]),
    "steak": np.array([-0.2, -0.45, 0.03, 0.707, 0.707, 0, 0]),
}

DEFAULT_OPENNESS_DICT = {
    "bucket_lift": 0,
    "bag_lift": 0,
    "bucket_move": 0,
    "bag_move": 0,
    "box_move": 0,
    "laptop_move": 0,
    "safe_move": 0,
    "suitcase_move": 0,
    "toaster_press": 0,
    "switch": 0,
    "stapler_press": 1.0,
    "stapler_move": 0,
}

DEFAULT_POSE = np.array([-0.2, -0.45, 0.03, 1, 0, 0, 0])
DEFAULT_OPENNESS = 0.5


def set_default_pose(obj, cls, id=None):
    if isinstance(obj, list):
        if id is None:
            for o, c in zip(obj, cls):
                set_default_pose(o, c)
        else:
            for o, c, i in zip(obj, cls, id):
                set_default_pose(o, c, i)
    else:
        pose = DEFAULT_POSE_DICT.get(cls, DEFAULT_POSE)
        if type(pose) == dict:
            pose = pose.get(id, DEFAULT_POSE)
        obj.random_pose = {"pos": [0, 0], "rot": 0}
        set_pose(obj, pose)


def set_default_openness(obj, cls, id=None):
    if isinstance(obj, list):
        for o, c, i in zip(obj, cls, id):
            set_default_openness(o, c, i)
    else:
        openness = DEFAULT_OPENNESS_DICT.get(cls, DEFAULT_OPENNESS)
        if type(openness) == dict:
            openness = openness.get(id, DEFAULT_OPENNESS)
        obj.random_openness = 0
        set_openness(obj, openness)


def get_random_pos(r):
    theta = np.random.uniform(0, 2 * np.pi)
    # Generate a random radius, properly scaled
    r = np.sqrt(np.random.uniform(0, r))
    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def set_random_pose(obj, cls, id=None, task="articulated"):
    if isinstance(obj, list):
        if id is None:
            for o, c in zip(obj, cls):
                set_random_pose(o, c)
        else:
            for o, c, i in zip(obj, cls, id):
                set_random_pose(o, c, i)
    else:
        pose = DEFAULT_POSE_DICT.get(cls, DEFAULT_POSE)
        if type(pose) == dict:
            pose = pose.get(id, DEFAULT_POSE)

        pos = pose[:3].copy()
        quat = pose[3:].copy()

        if task == "articulated" or task == "rigidbody" or cls in ALL_RIGIDBODY_OBJECTS:
            rand_xrange = 0.1
            rand_yrange = 0.1
            rand_pos = np.zeros(2)
            rand_pos[0] = np.random.uniform(-rand_xrange, rand_xrange)
            rand_pos[1] = np.random.uniform(-rand_yrange, rand_yrange)
            # rand_pos[0], rand_pos[1] = get_random_pos(0.13)
        elif task == "longhorizon" and cls in ALL_ARTICULATED_OBJECTS:
            rand_xrange = 0.05
            rand_yrange = 0.05
            rand_pos = np.zeros(2)
            rand_pos[0] = np.random.uniform(-rand_xrange, rand_xrange)
            rand_pos[1] = np.random.uniform(0, rand_yrange)

        pos[:2] += rand_pos

        if rand_pos[1] > 0:
            rand_rot = np.random.uniform(0, np.pi / 6)
        else:
            rand_rot = np.random.uniform(-np.pi / 6, 0)

        obj.random_pose = {"pos": rand_pos.tolist(), "rot": rand_rot}

        # Rotate the object around the z-axis
        if cls == "suitcase_rotate" or cls == "suitcase_move":
            quat = transforms3d.quaternions.qmult(quat, euler2quat(rand_rot, 0, 0))
        else:
            quat = transforms3d.quaternions.qmult(quat, euler2quat(0, 0, rand_rot))
        pose = np.concatenate([pos, quat])

        set_pose(obj, pose)


def set_random_openness(obj, cls, id):
    if isinstance(obj, list):
        for o, c, i in zip(obj, cls, id):
            set_random_openness(o, c, i)
    else:
        if type(DEFAULT_OPENNESS_DICT.get(cls, DEFAULT_OPENNESS)) == dict:
            openness = DEFAULT_OPENNESS_DICT.get(cls, DEFAULT_OPENNESS).get(
                id, DEFAULT_OPENNESS
            )
        else:
            openness = DEFAULT_OPENNESS_DICT.get(cls, DEFAULT_OPENNESS)

        randomness = np.random.uniform(-0.15, 0.15)
        openness += randomness
        openness = np.clip(openness, 0, 1)

        obj.random_openness = randomness

        set_openness(obj, openness)


def set_openness(obj, openness):
    if isinstance(obj, list):
        for ob in obj:
            set_openness(ob)
    else:
        obj.set_openness(openness=openness)


def set_pos(obj, value):
    if isinstance(obj, list):
        for ob in obj:
            set_pos(ob, value)
    else:
        obj.set_qpos(value)


def set_pose(obj, pose):
    """
    Set a given pose for the object.
    """

    if isinstance(obj, list):
        for ob in obj:
            set_pose(ob, pose)

    obj.set_pose(pose)


def get_distance(obj1, obj2):
    """
    Get the distance between two objects.
    """
    if isinstance(obj1, list):
        return [get_distance(ob1, ob2) for ob1, ob2 in zip(obj1, obj2)]
    else:
        obj1_pos = obj1.pos
        obj2_pos = obj2.pos
        distance = np.linalg.norm(obj1_pos - obj2_pos)
        return distance
