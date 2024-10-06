from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Optional

import gym
import icecream
import torch
import torch as th
from torch import nn
import torch.nn.functional as F

from gensim2.env.solver.rl.stable_baselines3.common.preprocessing import (
    get_flattened_obs_dim,
    is_image_space,
)
from gensim2.env.solver.rl.stable_baselines3.common.type_aliases import TensorDict
from gensim2.env.solver.rl.stable_baselines3.common.utils import get_device
from gensim2.env.solver.rl.stable_baselines3.networks.pointnet_modules.pointnet import (
    PointNet,
    PointNet_v1,
    PointNet_v2,
    # PointNet_v3,
    # PointNet_v4,
    # PointNet_v22,
    # PointNet_v23,
    # PointNet_v5,
    # PointNet_v52,
    # PointNet_v53,
    # PointNet_v54,
)
from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.gpnet import (
    GPNet_load1,
    GPNet_load2,
    GPNet,
    GPNet_v1,
    GPNet_v2,
    GPNet_v3,
    GPNet_v4,
    GPNet_v5,
    SegGPNet,
    SegGPNet_v1,
    SegGPNet_v2,
    SegGPNet_v3,
)


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = (
            []
        )  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = (
            []
        )  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(
                    nn.Linear(last_layer_dim_shared, layer)
                )  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(
                    layer, dict
                ), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(
                        layer["pi"], list
                    ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(
                        layer["vf"], list
                    ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(
            policy_only_layers, value_only_layers
        ):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int
                ), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int
                ), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def get_actor_critic_arch(
    net_arch: Union[List[int], Dict[str, List[int]]]
) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(
            net_arch, dict
        ), "Error: the net_arch can only contain be a list of ints or a dict"
        assert (
            "pi" in net_arch
        ), "Error: no key 'pi' was provided in net_arch for the actor network"
        assert (
            "qf" in net_arch
        ), "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


class PointNetExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        feat_key: Optional[str] = None,
        use_bn=True,
        local_channels=(64, 128, 256),
        global_channels=(256,),
        one_hot_dim=0,
        version=-1,
        gt_key: Optional[str] = None,
        gt_percent: Optional[float] = None,
    ):
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")
        if gt_key is not None:
            if gt_key not in list(observation_space.keys()):
                raise RuntimeError(
                    f"Ground truth key {gt_key} not in observation space."
                )

        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.has_gt = gt_key is not None
        self.gt_key = gt_key
        self.gt_percent = gt_percent
        pc_spec = observation_space[pc_key]
        pc_dim = pc_spec.shape[1]
        if self.has_feat:
            feat_spec = observation_space[feat_key]
            feat_dim = feat_spec.shape[1]
        else:
            feat_dim = 0
        features_dim = global_channels[-1]

        super().__init__(observation_space, features_dim)

        n_input_channels = pc_dim + feat_dim + one_hot_dim
        if version == -1:
            self.point_net = PointNet(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 1:
            self.point_net = PointNet_v1(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 2:
            self.point_net = PointNet_v2(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 3:
            self.point_net = PointNet_v3(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 4:
            self.point_net = PointNet_v4(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 22:
            self.point_net = PointNet_v22(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 23:
            self.point_net = PointNet_v23(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 5:
            self.point_net = PointNet_v5(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 52:
            self.point_net = PointNet_v52(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 53:
            self.point_net = PointNet_v53(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
            )
        elif version == 54:
            self.point_net = PointNet_v54(
                n_input_channels,
                local_channels=local_channels,
                global_channels=global_channels,
                use_bn=use_bn,
                gt_percent=gt_percent,
            )
        else:
            raise NotImplementedError(f"Unknown version of pointnet {version}")
        self.n_input_channels = n_input_channels
        self.n_output_channels = self.point_net.out_channels

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = torch.transpose(observations[self.pc_key], 1, 2)
        if self.has_feat:
            feats = torch.transpose(observations[self.feat_key], 1, 2)
        else:
            feats = None
        if self.has_gt:
            gt = observations[self.gt_key]
        else:
            gt = None
        # pointnet return {"feature": output_feature, "max_indices": max_indices}
        return self.point_net(points, points_feature=feats, gt=gt)["feature"]


class PointNetExtractorGP(BaseFeaturesExtractor):
    """
    :param observation_space:
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        feat_key: Optional[str] = None,
        emb_dim=128,
        groups=4,
        out_channel=128,
        local_channels=(64, 128, 256),
        global_channels=(256,),
        one_hot_dim=0,
        pn_only=False,
        gt_key: Optional[str] = None,
        gt_percent: Optional[float] = None,
    ):
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")
        if gt_key is not None:
            if gt_key not in list(observation_space.keys()):
                raise RuntimeError(
                    f"Ground truth key {gt_key} not in observation space."
                )

        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.has_gt = gt_key is not None
        self.gt_key = gt_key
        self.gt_percent = gt_percent
        pc_spec = observation_space[pc_key]
        pc_dim = pc_spec.shape[1]
        if self.has_feat:
            feat_spec = observation_space[feat_key]
            feat_dim = feat_spec.shape[1]
        else:
            feat_dim = 0
        features_dim = global_channels[-1]

        super().__init__(observation_space, features_dim)

        # n_input_channels = pc_dim + feat_dim + one_hot_dim

        self.point_net = GPNet(
            in_channel=3,
            emb_dim=emb_dim,
            group_dim=groups,
            out_channel=out_channel,
            pointnet_only=pn_only,
        )

        # self.n_input_channels = n_input_channels
        self.n_output_channels = out_channel

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = observations[self.pc_key]
        if self.has_feat:
            feats = torch.transpose(observations[self.feat_key], 1, 2)
        else:
            feats = None
        if self.has_gt:
            gt = observations[self.gt_key]
        else:
            gt = None
        # pointnet return {"feature": output_feature, "max_indices": max_indices}
        return self.point_net(points, points_feature=feats, gt=gt)


class RGBExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        rgb_key: str,
        feat_key: Optional[str] = None,
        out_channel=256,
        extractor_name="naturecnn",
        gt_key: Optional[str] = None,
        state_key="state",
        freeze_type=0,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        *kwargs,
    ):
        self.use_state = state_key is not None
        self.state_key = state_key
        self.freeze_type = freeze_type  # 0 means no free , 1 means normal freeze, 2 means padding 0 to the 256 dim.
        print(f"extractor use state = {self.use_state}")
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {state_key} not in observation space: {observation_space}"
                )
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if rgb_key not in list(observation_space.keys()):
            raise RuntimeError(f"RGB key {rgb_key} not in observation space.")

        super().__init__(observation_space, out_channel)
        self.rgb_key = rgb_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.gt_key = gt_key
        from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.pretrain_nets import (
            ResNet18,
            ResNet50,
        )

        if extractor_name == "res18":
            self.extractor = ResNet18(output_channel=256, pretrain=None)
        elif extractor_name == "res18_IMAGENET1K":
            self.extractor = ResNet18(output_channel=256, pretrain="IMAGENET1K")
        elif extractor_name == "res18_R3M":
            self.extractor = ResNet18(output_channel=256, pretrain="R3M")
        elif extractor_name == "res50":
            self.extractor = ResNet50(output_channel=256, pretrain=None)
        elif extractor_name == "res50_IMAGENET1K":
            self.extractor = ResNet50(output_channel=256, pretrain="IMAGENET1K")
        elif extractor_name == "res50_R3M":
            self.extractor = ResNet50(output_channel=256, pretrain="R3M")
        else:
            raise NotImplementedError(
                f"Extractor {extractor_name} not implemented. Available:\
            vgg mae"
            )

        self.n_output_channels = out_channel
        assert self.n_output_channels == 256

        # print the model
        # print(self.extractor)
        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = out_channel + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(
                *create_mlp(
                    self.state_dim, output_dim, net_arch, state_mlp_activation_fn
                )
            )

    def forward(self, observations: TensorDict) -> th.Tensor:
        # get rgb
        rgb = observations[self.rgb_key]  # B * 224 * 224 * 3
        b, _, _, _ = rgb.shape
        rgb_feat = self.extractor(rgb)  # B * 256
        if self.freeze_type == 2:
            rgb_feat = torch.zeros(size=(b, 256)).to("cuda:0")
        if self.use_state:
            state_feat = self.state_mlp(observations[self.state_key])
            return torch.cat([rgb_feat, state_feat], dim=-1)  # B * (256 + 64)
        else:
            return rgb_feat  # B * 256


class PointNetImaginationExtractorGP(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        feat_key: Optional[str] = None,
        out_channel=256,
        freeze_type=0,
        extractor_name="smallpn",
        gt_key: Optional[str] = None,
        imagination_keys=("imagination_robot",),
        state_key="state",
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        *kwargs,
    ):
        self.imagination_key = imagination_keys
        # Init state representation
        self.use_state = state_key is not None
        self.state_key = state_key
        self.freeze_type = freeze_type  # 0 means no free , 1 means normal freeze, 2 means padding 0 to the 256 dim.

        print(f"extractor use state = {self.use_state}")
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {state_key} not in observation space: {observation_space}"
                )
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")

        super().__init__(observation_space, out_channel)
        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.gt_key = gt_key

        # print(module_name, pn_only, use_history_obs)

        if extractor_name == "smallpn":
            from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.pretrain_nets import (
                PointNet,
            )

            self.extractor = PointNet()
        elif extractor_name == "mediumpn":
            from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.pretrain_nets import (
                PointNetMedium,
            )

            self.extractor = PointNetMedium()
        elif extractor_name == "largepn":
            from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.pretrain_nets import (
                PointNetLarge,
            )

            self.extractor = PointNetLarge()
        elif extractor_name == "largepn_half":
            from gensim2.env.solver.rl.stable_baselines3.networks.gpnet_modules.pretrain_nets import (
                PointNetLargeHalf,
            )

            self.extractor = PointNetLargeHalf()
        else:
            raise NotImplementedError(
                f"Extractor {extractor_name} not implemented. Available:\
             smallpn, mediumpn, largepn, largepn_half"
            )

        # self.n_input_channels = n_input_channels
        self.n_output_channels = out_channel
        assert self.n_output_channels == 256

        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = out_channel + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(
                *create_mlp(
                    self.state_dim, output_dim, net_arch, state_mlp_activation_fn
                )
            )

    def forward(self, observations: TensorDict) -> th.Tensor:
        # get raw point cloud segmentation mask
        points = observations[self.pc_key]  # B * N * 3
        b, _, _ = points.shape
        if len(self.imagination_key) > 0:
            for key in self.imagination_key:
                obs = observations[key]
                if len(obs.shape) == 2:
                    obs = obs.unsqueeze(0)
                img_points = obs[:, :, :3]
                points = torch.concat([points, img_points], dim=1)

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)  # B * 256
        if self.freeze_type == 2:
            pn_feat = torch.zeros(size=(b, 256)).to("cuda:0")
        if self.use_state:
            state_feat = self.state_mlp(observations[self.state_key])
            return torch.cat([pn_feat, state_feat], dim=-1)
        else:
            return pn_feat


def get_segmentation_feature(
    points,
    seg_type,
    points_before=None,
    gt_seg=None,
    semsegnet=None,
    gt_percent=0.0,
    confusion_percent=0.0,
):
    """
    Input:
        points: torch.tensor (B, N， 3)
        points_before: None(for single frame module) or torch.tensor (B, N， 3) (for 2 frame module)
        gt_seg: torch.tensor (B, N, 4)
    Return:
        feats: torch.tensor (B, 4, N)
    """
    assert seg_type in ["raw_pc", "inference_seg", "gt_seg"]
    # if this input is not batched, add batch dimension
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if len(gt_seg.shape) == 2:
        gt_seg = gt_seg.unsqueeze(0)
    points = torch.transpose(points, 1, 2)  # B * 3 * N
    batch_size, _, num_points = points.shape
    if seg_type == "raw_pc":
        feats = torch.transpose(gt_seg, 1, 2)  # B * 4 * N (all zeros)
    elif seg_type == "gt_seg":
        gt = torch.transpose(gt_seg, 1, 2)  # B * 4 * N (all gt)
        feats = torch.zeros(size=gt.shape, dtype=gt.dtype).to(gt.device)
        # confusion_element = torch.randint(low=0, high=5, size=(batch_size, num_points))  # B * N
        # feats[confusion_element]
        num_gt = int(gt_percent * num_points)
        # num_confusion = int(confusion_percent * (num_points - num_gt))
        idx = torch.randperm(num_points, dtype=torch.long)
        gt_idx = idx[:num_gt]  # (num_gt, )
        # confusion_idx = idx[num_gt: num_gt + num_confusion]
        confusion_element = torch.randint(
            low=0, high=5, size=(batch_size, num_points)
        )  # B * N
        feats[:, :, gt_idx] = gt[:, :, gt_idx]
        # feats[:, :, confusion_idx] =
    elif seg_type == "inference_seg":
        if semsegnet.frame == 1:
            seg_pred, trans_feat = semsegnet(points)
        elif semsegnet.frame == 2:
            seg_pred = semsegnet(
                pc1=points, pc2=points
            )  # [baochen]todo: which is the former one ?
        seg_pred = torch.argmax(
            seg_pred, 2, keepdim=False
        )  # [B, N], the max class index of each point
        if gt_percent > 0.0:
            gt = torch.transpose(gt_seg, 1, 2)  # B * 4 * N
            seg_gt = torch.argmax(gt, 1)
            num_gt = int(gt_percent * num_points)
            idx = torch.randperm(num_points)[:num_gt]
            seg_pred[:, idx] = seg_gt[:, idx]
        feats = torch.nn.functional.one_hot(seg_pred, num_classes=4)  # B * N * 4
        feats = torch.transpose(feats, 1, 2)  # B * 4 * N
        # seg_pred_0 = torch.reshape(torch.tensor(seg_pred == 0, dtype=torch.float32),
        #                            shape=(seg_pred.shape[0], 1, seg_pred.shape[1]))
        # seg_pred_1 = torch.reshape(torch.tensor(seg_pred == 1, dtype=torch.float32),
        #                            shape=(seg_pred.shape[0], 1, seg_pred.shape[1]))
        # seg_pred_2 = torch.reshape(torch.tensor(seg_pred == 2, dtype=torch.float32),
        #                            shape=(seg_pred.shape[0], 1, seg_pred.shape[1]))
        # seg_pred_3 = torch.reshape(torch.tensor(seg_pred == 3, dtype=torch.float32),
        #                            shape=(seg_pred.shape[0], 1, seg_pred.shape[1]))
        # feats = torch.cat([seg_pred_0, seg_pred_1, seg_pred_2, seg_pred_3], dim=1)  # B * 4 * N
        # gt = torch.transpose(gt_seg, 1, 2)  # B * 4 * N
        # feats = gt
    feats.requires_grad = False
    return points, feats  # B * 4 * N


class PointNetImaginationExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        feat_key: Optional[str] = None,
        use_bn=True,
        local_channels=(64, 128, 256),
        global_channels=(256,),
        gt_key: Optional[str] = None,
        imagination_keys=("imagination_robot",),
        state_key="state",
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        *kwargs,
    ):
        self.imagination_key = imagination_keys
        # Init state representation
        self.use_state = True
        self.state_key = state_key
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {state_key} not in observation space: {observation_space}"
                )
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")

        feat_dim = 0
        features_dim = global_channels[-1]
        super().__init__(observation_space, features_dim)
        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.has_gt = gt_key is not None
        self.gt_key = gt_key
        pc_spec = observation_space[pc_key]
        pc_dim = pc_spec.shape[1]

        n_input_channels = pc_dim
        self.point_net = PointNet(
            n_input_channels,
            local_channels=local_channels,
            global_channels=global_channels,
            use_bn=use_bn,
        )

        self.n_input_channels = n_input_channels
        self.n_output_channels = self.point_net.out_channels
        # State MLP
        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = self.point_net.out_channels + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(
                *create_mlp(
                    self.state_dim, output_dim, net_arch, state_mlp_activation_fn
                )
            )

    def forward(self, observations: TensorDict) -> th.Tensor:
        # get raw point cloud segmentation mask
        points = observations[self.pc_key]  # B * N * 3
        points = torch.transpose(points, 1, 2)
        # get imagination point cloud segmentation mask
        if len(self.imagination_key) > 0:
            for key in self.imagination_key:
                img_points_and_feature = observations[key].transpose(1, 2)  # B * 7 * Ni
                img_points = img_points_and_feature[:, :3, :]
                points = torch.concat([points, img_points], dim=2)
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.point_net(points, None)["feature"]
        state_feat = self.state_mlp(observations[self.state_key])
        return torch.cat([pn_feat, state_feat], dim=-1)


class PointNetImaginationExtractor2Frame(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        feat_key: Optional[str] = None,
        use_bn=True,
        local_channels=(64, 128, 256),
        global_channels=(256,),
        one_hot_dim=0,
        version=-1,
        gt_key: Optional[str] = None,
        gt_percent: Optional[float] = None,
        imagination_keys=("imagination_robot",),
        state_key="state",
        seg_type="raw_pc",
        module_name=None,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        *kwargs,
    ):
        self.imagination_key = imagination_keys
        self.seg_type = seg_type
        self.gt_percent = gt_percent
        assert self.seg_type in ["raw_pc", "inference_seg", "gt_seg"]
        # Init state representation
        self.use_state = state_key is not None
        self.state_key = state_key
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {state_key} not in observation space: {observation_space}"
                )
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")

        feat_dim = 0
        features_dim = global_channels[-1]
        super().__init__(observation_space, features_dim)
        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.has_gt = gt_key is not None
        self.gt_key = gt_key
        self.gt_percent = gt_percent
        self.model_name = module_name
        pc_spec = observation_space[pc_key]
        pc_dim = pc_spec.shape[1]

        n_input_channels = pc_dim + 4
        self.flow_net = FlowNet(4)

        self.n_input_channels = n_input_channels
        self.n_output_channels = 256

    def forward(self, observations: TensorDict) -> th.Tensor:
        # get raw point cloud segmentation mask
        points = observations[self.pc_key]  # B * N * 3
        previous_points = observations["previous-" + self.pc_key]
        pn_feat = self.flow_net(pc1=previous_points, pc2=points)["feature"]
        return pn_feat


class PointNetStateExtractor(PointNetExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_key: str,
        use_bn=True,
        local_channels=(64, 128, 256),
        global_channels=(256,),
        state_key="state",
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
    ):
        self.state_key = state_key
        if state_key not in observation_space.spaces.keys():
            raise RuntimeError(
                f"State key {state_key} not in observation space: {observation_space}"
            )
        self.state_space = observation_space[self.state_key]

        super().__init__(
            observation_space,
            pc_key,
            None,
            use_bn=use_bn,
            local_channels=local_channels,
            global_channels=global_channels,
            one_hot_dim=0,
        )

        self.state_dim = self.state_space.shape[0]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = self.point_net.out_channels + output_dim
        self._features_dim = self.n_output_channels
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_dim, output_dim, net_arch, state_mlp_activation_fn)
        )

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = torch.transpose(observations[self.pc_key], 1, 2)
        if self.has_feat:
            feats = torch.transpose(observations[self.feat_key], 1, 2)
        else:
            feats = None
        pn_feat = self.point_net(points, feats)["feature"]
        state_feat = self.state_mlp(observations[self.state_key])
        return torch.cat([pn_feat, state_feat], dim=-1)
