import icecream
from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.create_task import create_gensim

import numpy as np
import sapien.core as sapien

from common_parser import parser
from PIL import Image

from datetime import datetime

parser.add_argument("--qpos", action="store_true")
parser.add_argument("--traj", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    env = create_gensim(
        task_name="TestArm",
        sim_type="Sapien",
        use_gui=args.render,
        use_ray_tracing=args.rt,
        eval=False,
        asset_id=args.asset_id,
        cam=args.cam,
    )

    # test joint positions
    if args.qpos:
        env.task.env.agent.set_qpos(
            np.array(
                [
                    [
                        0.01231998,
                        0.09696837,
                        0.09863646,
                        -2.29102019,
                        -0.03313249,
                        2.54783339,
                        0.91524211,
                        0.04,
                        0.04000017,
                    ]
                ]
            )
        )

    # test solve trajecotry
    if args.traj:
        traj = np.loadtxt("traj.txt", delimiter=",")  # (n, 9) numpy array

    env.render()
    while not env.viewer.closed:
        # test keypoints
        if env.viewer.window.key_down("p"):
            rgba = env.viewer.window.get_float_texture("Color")
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            rgb_pil = Image.fromarray(rgb_img)
            t = datetime.now().strftime("%Y%m%d%H%M%S")
            rgb_pil.save(f"screenshot_{t}.png")
        env.render()

        # test solve trajectory
        if args.traj:
            n = traj.shape[0]
            interval = 100
            for i in range(interval * n):
                if i % interval == 0:
                    env.task.env.agent.set_qpos(traj[i // interval])
                env.render()
