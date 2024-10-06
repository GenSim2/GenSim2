import yaml
import json

yaml_path = "gensim2/env/solver/kpam/config/OpenSth.yaml"

yaml_content = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)

json_path = yaml_path.replace(".yaml", ".json")

with open(json_path, "w") as f:
    json.dump(yaml_content, f, indent=4)
