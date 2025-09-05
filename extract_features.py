"""
Main
-Process the yml config file
-Create an agent instance
-Run the agent
"""
from src.agents import *
from src.utils.utils import (
    updated_config,
    dict_print,
    create_save_loc,
    set_logger,
    save_configs,
    backup_code,
    set_seed,
)
import wandb

if __name__ == "__main__":
    # ############# handling the bash input arguments and yaml configuration file ###############
    config = updated_config()

    # create saving location and document config files
    create_save_loc(config) 
    save_dir = config["save_dir"]

    # document config files
    save_configs(config)
    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["train"]["seed"])
    set_logger(save_dir, config["log_level"], "train", config["comment"])
    backup_code(save_dir)

    # printing the configuration
    dict_print(config)

    # ############# Wandb setup ###############
    wandb.init(
        project="PROJECT", # TODO: update
        config=config,
        name=None if config["run_name"] == "" else config["run_name"],
        entity="ENTITY",  # TODO: your wandb username or team name
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled" 
        notes=config["save_dir"],  # to know where the model is saved!
    )

    # ############# agent setup ###############
    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)

    agent.run_extract_features()
    agent.finalize()
