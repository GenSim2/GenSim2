import os
from collections import OrderedDict

import numpy as np
import os
import numpy as np
import random

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import re
import openai
import IPython
import time
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import copy
import re
import random
import json
import operator
import csv
import itertools
import base64
import webcolors
import sapien.core as sapien
from sapien.utils import Viewer
from PIL import Image
import numpy as np
import os
from pydrake.geometry import (
    StartMeshcat,
)
from .meshcat_cpp_utils import draw_point_cloud
from pydrake.visualization import ModelVisualizer
import open3d as o3d
import matplotlib.pyplot as plt
import io
import yaml
import ast

from gensim2.pipeline.utils.open3d_RGBD import getOpen3DFromTrimeshScene
from gensim2.pipeline.utils.trimesh_render import lookAt
from gensim2.pipeline.utils.trimesh_URDF import getURDF


model = "gpt-4-turbo"
reply_max_tokens = 6000


def render_object_meshcat(urdf_path, keypoint_dict=None, vis=True, multiview=False):
    # add spheres as keypoints
    meshcat = StartMeshcat()
    meshcat.ResetRenderMode()
    meshcat.SetProperty("/Background", "visible", False)
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.5)
    meshcat.SetCameraPose([0.3, 0.3, 0], [0, 0, 0])

    if keypoint_dict is not None:
        points = []
        colors = []
        for color_name, point_loc in keypoint_dict.items():
            points.append(point_loc)
            color = webcolors.name_to_rgb(color_name)
            color = (
                np.array([color.red, color.green, color.blue], dtype=np.float32) / 255.0
            )
            colors.append(color)

        # the way to get an image
        draw_point_cloud(meshcat, "kepoint", np.array(points), np.array(colors))

    visualizer = ModelVisualizer(meshcat=meshcat)
    model = visualizer.parser().AddModels(urdf_path)
    IPython.embed()
    visualizer.Run(loop_once=True)

    # get image


def get_default_sapien_scene_camera():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.1, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(40),
        near=near,
        far=far,
    )

    # camera.set_pose(sapien.Pose(p=[1, 0, 0]))
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera.set_parent(parent=camera_mount_actor, keep_pose=False)

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-0.15, -0.15, 0.15])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    return scene, camera


def render_urdf_sapien(
    urdf_path,
    obj_path,
    keypoint_dict=None,
    return_depth=False,
    vis=True,
    multiview=False,
):
    # get the bounding box of the object using open3d

    if obj_path is None:
        urdf, controller = getURDF(urdf_path)

        # Load the mesh
        trimesh_scene = urdf.getMesh()
        mesh = getOpen3DFromTrimeshScene(trimesh_scene)
    else:
        mesh = o3d.io.read_triangle_mesh(obj_path)
    # Compute the axis-aligned bounding box
    aabb = mesh.get_axis_aligned_bounding_box()

    # Accessing bounding box properties
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound

    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound

    # add spheres as keypoints
    # keypoints = urdf_path.replace(".urdf", "_info.json")
    # scene, camera = get_default_sapien_scene_camera()

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    # scene.add_ground(-0.8)
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    asset = loader.load_kinematic(urdf_path)
    assert asset, "URDF not loaded."
    actor_builder = scene.create_actor_builder()

    scene.set_ambient_light([0.5, 0.5, 0.5])
    # scene.set_environment_map("assets/misc/ktx/day.ktx")
    scene.add_directional_light([0, 1, -1], [1.0, 1.0, 1.0], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.01, 300
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    # Calculate the distance needed to fit the bounding box in the viewport
    distance = max(bbox_extent) / (2 * np.tan(np.deg2rad(35) / 2))

    camera.set_pose(sapien.Pose(p=[0, 0, 0]))
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera.set_parent(parent=camera_mount_actor, keep_pose=False)

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    # camera pose 0
    cam_pos = np.array(bbox_center - [distance, 0, 0])
    cam_pos[0] = min(cam_pos[0], -near - 0.05)
    cam_pos[1] = cam_pos[0]
    cam_pos[2] = -cam_pos[0]

    cam_poses = [copy.deepcopy(cam_pos)]

    # camera pose 1
    cam_pos = np.array(bbox_center - [distance, 0, 0])
    cam_pos[0] = min(cam_pos[0], -near - 0.05)
    cam_pos[1] = cam_pos[0]
    cam_pos[2] = cam_pos[0]

    cam_poses.append(copy.deepcopy(cam_pos))

    # camera pose 2
    cam_pos = np.array(bbox_center - [distance, 0, 0])
    cam_pos[0] = min(cam_pos[0], -near - 0.05)
    cam_pos[1] = -cam_pos[0]
    cam_pos[2] = cam_pos[0]

    cam_poses.append(copy.deepcopy(cam_pos))

    # camera pose 3
    cam_pos = np.array(bbox_center + [distance, 0, 0])
    cam_pos[0] = -min(-cam_pos[0], near + 0.05)
    cam_pos[1] = cam_pos[0]
    cam_pos[2] = cam_pos[0]

    cam_poses.append(copy.deepcopy(cam_pos))

    index = 0
    rgba_pils = []
    for cam_pos in cam_poses:

        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

        boxes = []
        if keypoint_dict is not None:
            for idx, (color_name, keypoint_loc) in enumerate(keypoint_dict.items()):
                color = webcolors.name_to_rgb(color_name)
                color = (
                    np.array([color.red, color.green, color.blue], dtype=np.float32)
                    / 255.0
                )
                actor_builder.add_sphere_visual(radius=0.05, color=color)
                # actor_builder.add_box_collision(half_size=[1e-5, 1e-5, 1e-5])
                box = actor_builder.build(name=f"box_{idx}")  # Add a colored box
                box.set_pose(sapien.Pose(p=keypoint_loc))
                boxes.append(box)

        scene.step()  # make everything set
        scene.update_render()
        camera.take_picture()

        if return_depth:
            return camera.get_float_texture("Position")

        rgba = camera.get_float_texture("Color")[..., :3]  # [H, W, 4]

        # An alias is also provided
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")

        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(f"color{index}.png")  # camera pose 0
        index += 1

        for box in boxes:
            # remove the actor
            scene.remove_actor(box)
        rgba_pils.append((rgba_pil))

    # take 4 images and put them together in one image 2x2 grid
    rgba_pil = Image.new("RGB", (rgba_pils[0].width * 2, rgba_pils[0].height * 2))
    rgba_pil.paste(rgba_pils[0], (0, 0))
    rgba_pil.paste(rgba_pils[1], (rgba_pils[0].width, 0))
    rgba_pil.paste(rgba_pils[2], (0, rgba_pils[0].height))
    rgba_pil.paste(rgba_pils[3], (rgba_pils[0].width, rgba_pils[0].height))

    rgba_pil.save(f"color.png")  # combined camera poses
    return [(rgba_pil)]


def index_keypoints(keypoint_json, keypoint_color):
    keypoint_list = []
    for color in keypoint_color:
        if color in keypoint_json:
            keypoint_list.append(keypoint_json[color])
    return keypoint_list


def set_llm_model(model_name):
    global model, reply_max_tokens
    model = model_name

    print("using gpt model:", model)
    if model_name == "gpt-4o":  # longer context
        reply_max_tokens = 100 * 1000


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_text(folder, name, out):
    mkdir_if_missing(folder)
    with open(os.path.join(folder, name + ".txt"), "w") as fhandle:
        fhandle.write(out)


def save_code(folder, name, out, rename=False):
    mkdir_if_missing(folder)
    if rename:
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    with open(os.path.join(folder, name + ".py"), "w") as fhandle:
        fhandle.write(out)


def add_to_txt(full_interaction, message, with_print=False):
    """Add the message string to the full interaction"""
    full_interaction.append("\n\n" + message)
    if with_print:
        print("\n\n" + message)
    return full_interaction


def get_task_import_str():
    return (
        "import numpy as np\n"
        + "import sapien.core as sapien\n"
        + "import transforms3d\n"
        + "from gensim2.env.base.base_task import GenSimBaseTask\n"
        + "from gensim2.env.utils.rewards import *\n"
        + "import gym\n"
        + "from gensim2.env.utils.pose import *\n"
        + "from gensim2.env.utils.rewards import l2_distance_reward, alignment_reward, progress_reward, check_qpos\n"
    )


def extract_code(res):
    """parse code block"""
    # Pattern to find string between ```
    pattern = r"```(.*?)```"

    # Use re.findall to get all substrings within ```
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return "", ""

    code_string = code_string[0]
    code_string = code_string.replace("python", "")
    code_lines = code_string.split("\n")

    if "python" in code_string:
        code_lines = code_lines[1:]  # skip the first line

    class_def = [line for line in code_lines if line.startswith("class")]
    task_name = class_def[0]
    task_name = task_name[
        task_name.find("class ") : task_name.rfind("(GenSimBaseTask)")
    ][6:]

    print("task_name:", task_name)
    return get_task_import_str() + "\n".join(code_lines).strip(), task_name


def extract_reward(res):
    pattern = r"```(.*?)```"

    # Use re.findall to get all substrings within ```
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return "", ""

    print(code_string)

    code_string = code_string[0]
    code_string = code_string.replace("python", "")
    code_lines = code_string.split("\n")
    # for line in code_lines:
    #     line = '    ' + line

    if "python" in code_string:
        code_lines = code_lines[1:]  # skip the first line

    return "    " + "\n    ".join(code_lines).strip()


def extract_dict(res, prefix="new_task"):
    """parse task dictionary"""
    pattern = r"{(.*?)}"
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return ""

    code_string = code_string[0]
    code_string = code_string.replace("python", "")

    return prefix + "={" + code_string.replace("\n", "").strip() + "}"


def extract_list(res, prefix="code_reference"):
    """parse task dictionary"""
    pattern = r"\[(.*?)\]"
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return ""

    code_string = code_string[0]
    return prefix + "=[" + code_string.strip() + "]"


def extract_keypoints(res, prefix="keypoint_name_list"):
    """extract keypoints. format expected is 'keypoint_name_list'=[red, blue, green]"""
    pattern = r"\[(.*?)\]"
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return ""

    code_string = code_string[0]
    return ast.literal_eval("[" + code_string.strip() + "]")


def extract_assets(res):
    """parse generated assets"""
    pattern = r"<?xml(.*?)</robot>"
    code_string = re.findall(pattern, res, re.DOTALL)

    assets_pattern = r'robot name="(.*?)">'
    assets_string = re.findall(assets_pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return {}

    try:
        new_urdf = OrderedDict()
        for asset_path, code in zip(assets_string, code_string):
            new_urdf[asset_path] = "<?xml" + code

        # new_urdf_cmd ='new_urdf={' + code_string[0].rstrip() + '}'
        # exec(new_urdf_cmd)
        return new_urdf

    except:
        print("asset creation failure")
        print(str(traceback.format_exc()))
        return None


def extract_opt_config(res):
    """parse the optimization config for the keypoint solver"""
    pattern = r"```(.*?)```"

    # Use re.findall to get all substrings within ```
    code_strings = re.findall(pattern, res, re.DOTALL)
    if len(code_strings) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return dict()

    configs = []
    for code_string in code_strings:
        if "yaml" in code_string:
            code_string = code_string.replace("yaml", "")
            code_string = code_string.replace("python", "")
            configs.append(yaml.safe_load(code_string))

    return configs


def extract_multiple_tasks(res):
    """parse multiple tasks"""
    pattern = r"{(.*?)}"
    code_strings = re.findall(pattern, res, re.DOTALL)
    if len(code_strings) == 0:
        return ""

    tasks = []
    for code_string in code_strings:
        code_string = code_string.replace("python", "")
        code_string = code_string.replace("\n", "").strip()
        code_string = "{" + code_string + "}"

        tasks.append(eval(code_string))

    return tasks


def save_stat(
    cfg, output_dir, env_names, syntax_rate, run_rate, env_success_rate, diversity_score
):
    """save run results"""
    print("=========================================================")
    print(
        f"{cfg['prompt_folder']} | TOTAL SYNTAX_PASS_RATE: {syntax_rate * 100:.1f}% RUNTIME_PASS_RATE: {run_rate * 100:.1f}% ENV_PASS_RATE: {env_success_rate * 100:.1f}% DIVERSITY SCORE: {diversity_score:.3f}"
    )
    print("=========================================================")

    with open(os.path.join(output_dir, "eval_results.csv"), "w") as f:
        writer = csv.writer(f)
        row_info_name = ["prompt", "metric", "success"]
        writer.writerow(row_info_name)
        for col, stat in zip(
            ["syntax", "runtime", "origin_tasks. completion", "diversity"],
            [syntax_rate, run_rate, env_success_rate, diversity_score],
        ):
            row_info = [cfg["prompt_folder"], col, stat]
            writer.writerow(row_info)


def format_dict_prompt(task_name_dict, sample_num=-1, sort_items=False):
    """format a saved dictionary into prompt"""
    if sort_items:
        task_name_dict = sorted(task_name_dict.items(), key=operator.itemgetter(0))
    prompt_replacement = ""
    sample_idx = list(range(len(task_name_dict)))
    random.shuffle(sample_idx)

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_name_dict), sample_num, replace=False)

    for idx, (task_name, task_desc) in enumerate(task_name_dict.items()):
        if idx in sample_idx:
            prompt_replacement += f"- {task_name}: {task_desc}\n"

    return prompt_replacement + "\n\n"


def format_keypoint_prompt(keypoint_dict):
    formatted_string = ""
    for key, value in keypoint_dict.items():
        formatted_string += f"'{key}': '{value}'\n"
    return formatted_string


def format_list_prompt(task_list, sample_num=-1, sort_items=False):
    """format a saved dictionary into prompt"""

    # if sort_items:
    #     task_list = sorted(task_list, key=operator.itemgetter(0))
    prompt_replacement = ""
    sample_idx = list(range(len(task_list)))

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_list), sample_num, replace=False)

    for idx, task in enumerate(task_list):
        if idx in sample_idx:
            prompt_replacement += (
                f"- {task['task-name']}: {task['task-descriptions']}\n"
            )

    return prompt_replacement + "\n\n"


def sample_list_reference(item_list, sample_num=-1):
    """sample reference code from a list of python files"""
    sample_idx = list(range(len(item_list)))
    prompt_replacement = ""

    if sample_num > 0:
        sample_idx = np.random.choice(len(item_list), sample_num, replace=False)

    print("reference files: ", [item_list[idx] for idx in sample_idx])
    for idx, item in enumerate(item_list):
        try:
            item_content = open(f"cliport/origin_tasks/{item}").read()
        except:
            # one or the other
            item_content = open(f"cliport/generated_tasks/{item}").read()

        if idx in sample_idx:
            prompt_replacement += f"```\n{item_content}\n```\n\n"

    return prompt_replacement + "\n\n"


def compute_diversity_score_from_assets_old(task_assets):
    """compute how many new asset combos are covered by previous by a proxy"""
    if len(task_assets) == 0:
        return 0

    existing_assets = []
    for asset in task_assets:
        new_asset_flag = True
        for existing_asset in existing_assets:
            # it's covered by any previous assets
            if set(asset).issubset(existing_asset):
                new_asset_flag = False
                break

        if new_asset_flag:
            existing_assets.append(asset)

    return len(existing_assets) / len(task_assets)


def iou_assets(asset1, asset2):
    asset1 = set(asset1)
    asset2 = set(asset2)
    return len(asset1 & asset2) / len(asset1 | asset2)


def compute_diversity_score_from_assets(task_assets, total_trials):
    """compute the pairwise IOU for assets"""
    if len(task_assets) == 0:
        return 0

    score = 0
    pairs = list(itertools.combinations(range(len(task_assets)), 2))
    for j, k in pairs:
        score += 1.0 - iou_assets(task_assets[j], task_assets[k])

    return score / (len(pairs) + 1e-6)


def truncate_message_for_token_limit(message_history, max_tokens=6000):
    truncated_messages = []
    tokens = 0

    # reverse
    for idx in range(len(message_history) - 1, -1, -1):
        message = message_history[idx]
        message_tokens = len(message["content"]) / 4  # rough estimate.
        if tokens + message_tokens > max_tokens:
            break  # This message would put us over the limit

        truncated_messages.append(message)
        tokens += message_tokens

    truncated_messages.reverse()
    return truncated_messages


def insert_system_message(message_history):
    system_message_prompt = "You are a helpful and expert assistant in robot simulation code writing and task design."
    "You design origin_tasks that are creative and do-able by table-top manipulation. "
    "You write code without syntax errors and always think through and document your code carefully. "
    message_history.insert(0, {"role": "system", "content": system_message_prompt})


# globally always feed the previous reply as the assistant message back into the model
existing_messages = []


def generate_feedback(
    prompt, max_tokens=2048, temperature=0.0, interaction_txt=None, retry_max=5, n=1
):

    global existing_messages, reply_max_tokens
    """use GPT-4 API"""
    existing_messages.append({"role": "user", "content": prompt})
    truncated_messages = truncate_message_for_token_limit(
        existing_messages, reply_max_tokens
    )

    if model.startswith("o1"):
        params = {
            "model": model,
            # "max_completion_tokens": max_tokens,
            "messages": truncated_messages,
            "n": n,
        }
    else:
        insert_system_message(truncated_messages)
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": truncated_messages,
            "n": n,
        }

    for retry in range(retry_max):
        try:
            if interaction_txt is not None:
                add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
            call_res = openai.ChatCompletion.create(**params)
            res = call_res["choices"][0]["message"]["content"]
            existing_messages.append({"role": "assistant", "content": res})

            to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if interaction_txt is not None:
                add_to_txt(interaction_txt, ">>> Answer: \n" + res, with_print=False)

            if n > 1:
                return [r["message"]["content"] for r in call_res["choices"]]
            return res

        except Exception as e:
            print("failed chat completion", e)
    raise Exception("Failed to generate")


def generate_feedback_visual(
    prompt,
    images,
    max_tokens=2048,
    temperature=0.0,
    interaction_txt=None,
    retry_max=5,
    n=1,
):
    """Assume images are in the format of list of numpy array"""

    global existing_messages, reply_max_tokens

    """GPT-4 Visual API."""
    curr_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

    for idx, image in enumerate(images):
        # cv2.imwrite(f"chat_{len(existing_messages)}_{idx}.png", msg)  # save as logs
        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")
        image_bytes = byte_stream.getvalue()
        curr_message["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                },
            }
        )
    existing_messages.append(curr_message)
    truncated_messages = truncate_message_for_token_limit(
        existing_messages, reply_max_tokens
    )
    insert_system_message(truncated_messages)

    params = {
        "model": "gpt-4-turbo",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": truncated_messages,
        "n": n,
    }

    # do not add image to the chat history for budget purpose
    existing_messages = existing_messages[:-1]

    for retry in range(retry_max):
        try:
            if interaction_txt is not None:
                add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)

            call_res = openai.ChatCompletion.create(**params)
            res = call_res["choices"][0]["message"]["content"]
            existing_messages.append({"role": "assistant", "content": res})

            to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if interaction_txt is not None:
                add_to_txt(
                    interaction_txt, ">>> (Visual) Answer: \n" + res, with_print=False
                )

            if n > 1:
                return [r["message"]["content"] for r in call_res["choices"]]
            return res

        except Exception as e:
            print("failed chat completion", e)
    raise Exception("Failed to generate")


def clear_messages():
    global existing_messages
    existing_messages = []
