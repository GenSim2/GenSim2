from gensim2.agent.utils.robot.utils import vis_pcd

import pickle

# Specify the path to your pkl file
file_path = "/media/ExtHDD/gensim_v2/data/close_box/episodes0/low_dim_obs.pkl"

# Open the pkl file in read mode
with open(file_path, "rb") as file:
    # Load the data from the pkl file
    data = pickle.load(file)

# Now you can work with the loaded data
# For example, you can print it
print(len(data))

print(data[0].keys())

print(type(data[0]["pointcloud"]))
print(data[0]["pointcloud"].keys())
vis_pcd(data[0]["pointcloud"], is_dict=True)
