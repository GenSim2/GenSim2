import numpy as np
import wandb
import os
import ipdb
import sapien
import logging
from wandb.sdk.lib import telemetry as wb_telemetry

from pathlib import Path
from typing import Optional, Callable, List

from gensim2.env.solver.rl.camera_utils import generate_imagination_pc_from_obs
from gensim2.env.solver.rl.stable_baselines3.common.callbacks import BaseCallback
from gensim2.env.solver.rl.video import VideoRecorder

logger = logging.getLogger(__name__)


class WorkspaceRL(BaseCallback):
    """Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 100,
        eval_freq: Optional[int] = None,
        eval_success_rate_freq: Optional[int] = None,
        eval_success_rate_times=1,
        eval_env_fn: Optional[Callable] = None,
        train_env_fn: Optional[Callable] = None,
        eval_cam_names: Optional[List[str]] = None,
        visualize_pc=True,
        gradient_save_freq: int = 0,
        rollout=0,
    ):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")

        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True

        self.model_save_freq = model_save_freq
        self.model_save_path = Path.cwd() / "model"
        self.visualize_pc = visualize_pc

        self.eval_freq = eval_freq
        self.eval_success_rate_freq = (
            eval_success_rate_freq  # the eval frequency of success rate
        )
        self.eval_success_rate_times = (
            eval_success_rate_times  # if 10, means eval each instance 10 times.
        )
        self.eval_env_fn = eval_env_fn
        self.train_env_fn = train_env_fn
        self.eval_cam_names = eval_cam_names

        self.gradient_save_freq = gradient_save_freq
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.roll_out = rollout

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def _on_rollout_end(self) -> None:
        need_restore = self.model.__dict__.get("need_restore", False)
        current_restore_step = self.model.__dict__.get("current_restore_step", 0)
        wandb.log({"rollout/restore": current_restore_step}, step=self.roll_out + 1)
        if need_restore and current_restore_step <= 5:
            return

        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.roll_out % self.model_save_freq == 0:
                    self.save_model()

        if self.eval_success_rate_freq is not None and self.eval_env_fn is not None:
            if self.roll_out % self.eval_success_rate_freq == 0 and self.roll_out > 0:
                # ================ calculate train env ========================
                train_env = self.train_env_fn()
                reward_sum = 0
                success_sum = 0
                # train_success_trajectory = len(train_env.instance_list) * self.eval_success_rate_times
                train_success_trajectory = self.eval_success_rate_times
                for i in range(train_success_trajectory):
                    obs = train_env.reset()
                    eval_success = False
                    for j in range(train_env.horizon):
                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                obs[key] = value[np.newaxis, :]
                        else:
                            obs = obs[np.newaxis, :]
                        action = self.model.policy.predict(
                            observation=obs, deterministic=True
                        )[0]
                        if len(action.shape) > 1:
                            action = action[0, :]
                        obs, reward, done, info = train_env.step(action)

                        train_env.scene.update_render()

                        if info["success"]:
                            eval_success = True

                        reward_sum += reward
                        if done:
                            break
                    success_sum += int(eval_success)

                wandb.log(
                    {"train/reward": reward_sum / train_success_trajectory},
                    step=self.roll_out + 1,
                )
                wandb.log(
                    {"train/succ": success_sum / train_success_trajectory},
                    step=self.roll_out + 1,
                )

                print("train mean reward = ", reward_sum / train_success_trajectory)
                print("train success rate = ", success_sum / train_success_trajectory)
                # ================ calculate test env ========================
                env = self.eval_env_fn()
                reward_sum = 0
                success_sum = 0
                # eval_success_trajectory = len(env.instance_list) * self.eval_success_rate_times
                eval_success_trajectory = self.eval_success_rate_times

                # stage = np.zeros(4, dtype=np.float64)
                for i in range(eval_success_trajectory):
                    obs = env.reset()
                    eval_success = False
                    for j in range(env.horizon):
                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                obs[key] = value[np.newaxis, :]
                        else:
                            obs = obs[np.newaxis, :]
                        action = self.model.policy.predict(
                            observation=obs, deterministic=True
                        )[0]
                        if len(action.shape) > 1:
                            action = action[0, :]
                        obs, reward, done, info = env.step(action)
                        # stage[env.state - 1] += 1
                        env.scene.update_render()

                        if info["success"]:
                            eval_success = True

                        reward_sum += reward
                        if done:
                            break
                    success_sum += int(eval_success)
                wandb.log(
                    {"val/reward": reward_sum / eval_success_trajectory},
                    step=self.roll_out + 1,
                )
                wandb.log(
                    {"val/succ": success_sum / eval_success_trajectory},
                    step=self.roll_out + 1,
                )

                print(f"unseen mean reward = {reward_sum / eval_success_trajectory}")
                print(f"unseen success rate = {success_sum / eval_success_trajectory}")

        if self.eval_freq is not None and self.eval_env_fn is not None:
            if self.roll_out % self.eval_freq == 0:  # and self.roll_out > 0:
                env = self.eval_env_fn()

                self.video_recorder = VideoRecorder(
                    self.model_save_path, render_size=256, fps=20
                )
                self.video_recorder.init(env)

                reward_sum = 0
                obs = env.reset()
                img_dict = {key: [] for key in self.eval_cam_names}
                acc_list = list()
                has_done = False

                cam_training = None
                if env.cameras.__contains__("instance_1"):
                    cam_training: Optional[sapien.core.CameraEntity] = env.cameras[
                        "instance_1"
                    ]

                for i in range(env.horizon):
                    if isinstance(obs, dict):
                        for key, value in obs.items():
                            obs[key] = value[np.newaxis, :]
                    else:
                        obs = obs[np.newaxis, :]
                    action = self.model.policy.predict(
                        observation=obs, deterministic=True
                    )[0]
                    if len(action.shape) > 1:
                        action = action[0, :]
                    # obs, reward, done, info = env.step(action, eval=i, eval_cam_names=self.eval_cam_names)
                    env.update_render()
                    obs, reward, done, info = env.step(
                        action, eval=1, eval_cam_names=self.eval_cam_names
                    )  # TODO need to add eval_cam_names parameter somehow?
                    self.video_recorder.record(env)

                    # env.scene.update_render()
                    # for cam_name in self.eval_cam_names:
                    #     cam: sapien.core.CameraEntity = env.cameras[cam_name]
                    #     if cam_training:
                    #         cam.set_local_pose(cam_training.get_pose())

                    #     # await_dl_list = cam.take_picture_and_get_dl_tensors_async(["Color"])  # how is this done?
                    #     # dl_list = await_dl_list.wait()
                    #     # dl_list = [cam.get_dl_tensor(name) for name in ["Color"]]
                    #     #
                    #     # dl_tensor = dl_list[0]
                    #     # import torch
                    #     # output_array = torch.from_dlpack(dl_tensor).cpu().numpy()
                    #     # output_array = output_array[..., :3]
                    #     # print(f"cam = {cam}")
                    #     cam.take_picture()
                    #     output_array = cam.get_color_rgba()
                    #     img_dict[cam_name].append(output_array)
                    #     # cam.take_picture()
                    #     # img_dict[cam_name].append(fetch_texture(cam, "Color", return_torch=False))
                    if not has_done:
                        reward_sum += reward
                    if done:
                        has_done = True
                        # if info['img_dict']:
                        # img_dict = info['img_dict']

                if len(acc_list) > 0:
                    wandb.log(
                        {"acc": np.mean(np.array(acc_list))}, step=self.roll_out + 1
                    )
                    print(f"accuracy = {np.mean(np.array(acc_list))}")

                print("Saving Video")

                saved_video = False
                for cam_name, img_list in self.video_recorder.frames.items():
                    if img_list is None:
                        print(f"img_list is None")
                        continue

                    try:
                        video_array = (np.stack(img_list, axis=0) * 255).astype(
                            np.uint8
                        )  # 0~1 -> 0~255
                    except:
                        ipdb.set_trace()
                    video_array = np.stack(img_list, axis=0).astype(np.uint8)
                    # # video_array = 255 - video_array
                    video_array = np.transpose(
                        video_array, (0, 3, 1, 2)
                    )  # tHWC -> tCHW
                    wandb.log(
                        {
                            f"{cam_name}": wandb.Video(
                                video_array,
                                fps=20,
                                format="mp4",
                                caption=f"Reward: {reward_sum:.2f}",
                            )
                        },
                        step=self.roll_out + 1,
                    )
                    saved_video = True

        self.current_restore_step = 0
        self.roll_out += 1

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{self.roll_out}")

        self.model.save(path)
        wandb.save(path + ".zip", base_path=self.model_save_path)

        # write the last checkpoint to last_checkpoint.txt
        with open(os.path.join(self.model_save_path, "last_checkpoint.txt"), "w") as f:
            f.write(path)
            print(f"Saved model to {path}")
        if self.verbose > 1:
            logger.info("Saving model checkpoint to " + path)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True
