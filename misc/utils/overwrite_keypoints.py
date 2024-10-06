import os
import json


dir = "assets/articulated_objs/suitcase_rotate_new/"

for id in os.listdir(dir):

    keypoint_path = f"{dir}/{id}/keypoints_final.json"
    target_keypoint_path = f"{dir}/{id}/info.json"

    if not os.path.exists(keypoint_path) or not os.path.exists(target_keypoint_path):
        print(f"keypoint file {keypoint_path} or {target_keypoint_path} not found")
        continue

    kp = json.load(open(keypoint_path))
    tkp = json.load(open(target_keypoint_path))

    tkp["keypoints"]["articulated_object_head"] = kp["keypoints"]["red"]
    tkp["keypoints"]["articulated_object_tail"] = kp["keypoints"]["yellow"]
    # tkp["keypoints"]["articulated_object_bottom_base"] = kp["keypoints"]["blue"]

    json.dump(tkp, open(target_keypoint_path, "w"), indent=4)
