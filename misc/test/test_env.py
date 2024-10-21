import icecream
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim

import numpy as np
import sapien.core as sapien

from scripts.common_parser import parser
from PIL import Image

from datetime import datetime

if __name__ == "__main__":
    args = parser.parse_args()

    env = create_gensim(
        task_name=args.env,
        sim_type="Sapien",
        use_gui=args.render,
        use_ray_tracing=args.rt,
        eval=False,
        asset_id=args.asset_id,
        cam=args.cam,
    )

    icecream.ic(env.horizon)
    
    builder = env.scene.create_actor_builder()
    
    obj_keypoints = env.get_object_keypoints()
    tool_keypoints = env.get_tool_keypoints()

    for p in obj_keypoints.values():
        builder.add_box_visual(half_size=[0.005, 0.005, 0.005], color=[1.0, 0.0, 0.0])
        box = builder.build(name="box")  # Add a box
        box.set_pose(sapien.Pose(p=p))

    for p in tool_keypoints.values():
        builder.add_box_visual(half_size=[0.005, 0.005, 0.005], color=[0.0, 1.0, 0.0])
        box = builder.build(name="box")
        box.set_pose(sapien.Pose(p=p))

    if args.render:
        env.render()

    env.reset(args.random)
    while args.render and not env.viewer.closed:
        # test keypoints
        if env.viewer.window.key_down("p"):
            rgba = env.viewer.window.get_float_texture("Color")
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            rgb_pil = Image.fromarray(rgb_img)
            t = datetime.now().strftime("%Y%m%d%H%M%S")
            rgb_pil.save(f"screenshot_{t}.png")

        if args.render:
            env.render()
