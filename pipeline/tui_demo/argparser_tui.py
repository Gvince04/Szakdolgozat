import os
import json

# CONFIGURATION
REGISTRY = "pipeline_registry.json"
ROOT = "../"

# FUNCTIONS
# Load pipeline from json
def load_pipeline(path):
    with open(path, "r") as f:
        return json.load(f)

# Advance pipeline based on json
def advance_pipeline(registry):
    print(registry["project_name"])

    for item in registry["steps"]:
        _id = item["id"]
        name = item["name"]
        script_path = item["script_path"]
        config_path = item["config_path"]

        print(name)
        
        solve_state(_id, name, script_path, config_path)
        print("-"*30)

def solve_state(_id, name, script_path, config_path):
    script_path = ROOT + script_path
    config_path = ROOT + config_path
    
    schema = load_pipeline(config_path)

    default_config = {}
    
    for item in schema:
        flag = item["flag"]
        name = item["name"]
        _type = item["type"]
        default = item["default"]
        tip = item["tip"]

        draw_tui(flag, name, _type, default, tip)

        default_config[flag] = default

    user_config = get_user_input(default_config)
    print_command(script_path, user_config, default_config)

def draw_tui(flag, name, _type, default, tip):
    tab = " "*4
    print(f'{tab}{name}: default="{default}" ({_type})')
    print(f"{tab+tab}({tip})")
    print()

def get_user_input(default_config):
    user_config = default_config.copy()
    line = ""
    while True:
        line = input().strip().split(',')
        if len(line) == len(default_config):
            break
    
    for i in range(len(line)):
        key = list(default_config.keys())[i]

        if line[i] != "":
            user_config[key] = line[i]

    return user_config

def print_command(script_path, user_config, default_config):
    script = script_path.split("/")[-1]

    print()
    print(f"```command: python {script} ", end="")

    for (key, user_val), (_, def_val) in zip(user_config.items(), default_config.items()):
        if user_val != def_val:
            print(f"{key} {user_val}", end=" ")
    print("```")

# MAIN
def main():
    registry = load_pipeline(REGISTRY)
    
    advance_pipeline(registry)


if __name__ == "__main__":
    main()