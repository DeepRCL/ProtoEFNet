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
    save_configs,
    set_logger,
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
        project="ProtoEF",
        config=config,
        name=None if config["run_name"] == "" else config["run_name"],
        entity="ENTITY", 
        mode=config["wandb_mode"],  # one of "online", "offline" or "disabled" 
        notes=config["save_dir"],  # to know where the model is saved!
    )

    # ############# agent setup ###############
    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)

    # ############# Run the system ###############
    if config["eval_only"]:
        agent.evaluate(mode=config["eval_data_type"])
    elif config["push_only"]:
        agent.push(replace_prototypes=False)
    # if push_at_end exists in confid["model"] keys and is True, then train_push_at_end
    elif config["model"].get("push_at_end", False): # false indicates default value if it doesnt exist in config file
        agent.train_push_at_end()
    elif config["model"].get("warm_only", False): # if warm_only exists in confid["model"] keys and is True, then train all but last layer
        agent.warm_only()
    elif config["model"].get("last_only", False):
        agent.last_only()
    else:
        agent.run()

    agent.finalize()