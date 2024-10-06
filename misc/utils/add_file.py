import os

dir = "assets/articulated_objs/suitcase_rotate_new/"
file_path = "assets/articulated_objs/suitcase_rotate/101668/info.json"

# for all folders under dir, if there is no info.json in that folder, copy one


for root, dirs, _ in os.walk(dir):
    for d in dirs:
        if not os.path.exists(os.path.join(root, d, "info.json")):
            os.system(f"cp {file_path} {os.path.join(root, d, 'info.json')}")
