"""
Main
-Process the yml config file
-Create an agent instance
-Run the agent
"""
from src.agents import *
from src.utils.utils import (
    updated_config,
    update_nested_dict,
    dict_print,
    create_save_loc,
    save_configs,
    set_logger,
    backup_code,
    set_seed,
)
import wandb
import logging


def train():
    # ############# handling the bash input arguments and yaml configuration file ###############
    config = updated_config()

    # create saving location
    create_save_loc(config)  # config['save_dir'] gets updated here!
    save_dir = config["save_dir"]

    # printing the configuration
    dict_print(config)

    # ############# Wandb setup ###############
    wandb.init(
        # project="AorticStenosis_XAI",  # IGNORED IN SWEEP
        config=config,
        name=None if config["run_name"] == "" else config["run_name"],
        # entity="rcl_stroke",  # your wandb username or team name
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled"  TODO ??  TODO Add the config!
        notes=config["save_dir"],  # to know where the model is saved!
    )
    # Update config based on wandb sweep selected configs
    config_wandb = wandb.config
    config = update_nested_dict(config, config_wandb)

    # printing the configuration again
    logging.info(f"################################ NEW CONFIGS ######################")
    dict_print(config)

    ################## document config files
    save_configs(config)
    # ############# handling the logistics of (seed), and (logging) ###############
    set_seed(config["train"]["seed"])
    set_logger(save_dir, config["log_level"], "train", config["comment"])
    backup_code(save_dir)


    # ############# agent setup ###############
    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)

    # ############# Run the system ###############
    # if push_at_end exists in confid["model"] keys and is True, then train_push_at_end
    if config["model"].get("push_at_end", False):
        agent.train_push_at_end()
    else:
        agent.run()
    agent.finalize()


if __name__ == "__main__":
    config = updated_config()
    wandb.agent(config['SWEEP_ID'], train, count=config['SWEEP_COUNT'])

