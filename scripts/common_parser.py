import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="OpenBox")
parser.add_argument("--asset_id", type=str, default="")
parser.add_argument("--obs_mode", type=str, default="state")
parser.add_argument("--random", action="store_true")
parser.add_argument("--num_pcd", type=int, default=1200)
parser.add_argument("--render", action="store_true")
parser.add_argument("--rt", action="store_true")
parser.add_argument("--cam", type=str, default="default")
parser.add_argument("--save", action="store_true")
