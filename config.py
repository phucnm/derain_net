import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--data_dir", type=str,
                       default="../train/data/",
                       help="Directory with input data")
train_arg.add_argument("--label_dir", type=str,
                       default="../train/gt/",
                       help="Directory with label data")
train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")
train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")
train_arg.add_argument("--num_epoch", type=int,
                       default=25,
                       help="Number of epochs to train")
train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")
model_arg.add_argument("--num_res_layer", type=int,
                       default=5,
                       help="Number of layers in a ResNet block")
model_arg.add_argument("--num_time_step", type=int,
                       default=4,
                       help="Number of attention maps")
model_arg.add_argument("--theta", type=int,
                       default=0.8,
                       help="Parameter to compute attentive loss")                       
model_arg.add_argument("--gamma", type=int,
                       default=0.05,
                       help="Parameter to compute discriminator loss")                       


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
