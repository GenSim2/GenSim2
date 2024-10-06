import os


dir = "assets/articulated_objs/faucet"

for d in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, d)):
        for f in os.listdir(os.path.join(dir, d)):
            if f == "keypoints.json":
                os.remove(os.path.join(dir, d, f))
                print("Deleted: " + os.path.join(dir, d, f))
        # for f in os.listdir(os.path.join(dir, d)):
        #     if f == "info_test.json":
        #         os.rename(os.path.join(dir, d, f), os.path.join(dir, d, "info.json"))
