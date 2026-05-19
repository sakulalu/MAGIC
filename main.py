import argparse
import os
import random
import warnings

import numpy as np
import torch

from dataprocessing import MultiviewData
from layers import MAGICNetwork
from loss import SemanticAlignmentLoss
from models import pre_train, contrastive_train

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC for incomplete multi-view clustering")

    parser.add_argument("--db", type=str, default="BDGP",
                        choices=["BDGP", "MNIST-USPS", "FMNIST", "Fashion"],
                        help="Dataset name.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device index.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--mse_epochs", type=int, default=None,
                        help="Number of reconstruction pretraining epochs.")
    parser.add_argument("--con_epochs", type=int, default=None,
                        help="Number of contrastive fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--js_weight", type=float, default=0.1,
                        help="Weight of the JS term in semantic alignment.")
    parser.add_argument("--load_model", action="store_true", help="Load a saved MAGIC model.")
    parser.add_argument("--save_model", action="store_true", help="Save the trained MAGIC model.")

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_config(db):
    configs = {
        "BDGP": {
            "learning_rate": 1e-4,
            "batch_size": 200,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 100,
            "temperature_l": 0.7,
            "normalized": True,
            "dim_high_feature": 2000,
            "dim_low_feature": 1024,
            "dims": [256, 512],
            "beta": 0.01,
            "gamma": 1.0,
        },
        "MNIST-USPS": {
            "learning_rate": 1e-4,
            "batch_size": 50,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 25,
            "temperature_l": 0.7,
            "normalized": False,
            "dim_high_feature": 1500,
            "dim_low_feature": 1024,
            "dims": [256, 512, 1024],
            "beta": 0.01,
            "gamma": 1.0,
        },
        "FMNIST": {
            "learning_rate": 1e-4,
            "batch_size": 200,
            "seed": 10,
            "mse_epochs": 100,
            "con_epochs": 100,
            "temperature_l": 0.7,
            "normalized": True,
            "dim_high_feature": 1024,
            "dim_low_feature": 1024,
            "dims": [256, 512, 1024],
            "beta": 0.01,
            "gamma": 1.0,
        },
        "Fashion": {
            "learning_rate": 1e-4,
            "batch_size": 100,
            "seed": 20,
            "mse_epochs": 100,
            "con_epochs": 20,
            "temperature_l": 0.5,
            "normalized": True,
            "dim_high_feature": 2000,
            "dim_low_feature": 500,
            "dims": [256, 512],
            "beta": 0.01,
            "gamma": 1.0,
        },
    }
    return configs[db]


def apply_overrides(config, args):
    for name in ["seed", "mse_epochs", "con_epochs", "batch_size", "learning_rate"]:
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    return config


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = apply_overrides(dataset_config(args.db), args)
    set_seed(config["seed"])

    print("==========")
    print("Method: MAGIC")
    print(f"Dataset: {args.db}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Pretraining epochs: {config['mse_epochs']}")
    print(f"Fine-tuning epochs: {config['con_epochs']}")
    print("==========")

    mv_data = MultiviewData(args.db, device)
    num_views = len(mv_data.data_views)
    num_clusters = np.unique(mv_data.labels).size
    input_sizes = np.array([view.shape[1] for view in mv_data.data_views], dtype=int)

    model = MAGICNetwork(
        num_views=num_views,
        input_sizes=input_sizes,
        dims=config["dims"],
        dim_high_feature=config["dim_high_feature"],
        dim_low_feature=config["dim_low_feature"],
        num_clusters=num_clusters,
    ).to(device)

    alignment_loss = SemanticAlignmentLoss(
        config["batch_size"],
        num_clusters,
        js_weight=args.js_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=args.weight_decay,
    )

    model_path = f"./models/MAGIC_pytorch_model_{args.db}.pth"

    if args.load_model:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        pre_train(model, mv_data, config["batch_size"], config["mse_epochs"], optimizer)

        for epoch in range(config["con_epochs"]):
            model.set_aug_strength(epoch)
            contrastive_train(
                model,
                mv_data,
                alignment_loss,
                config["batch_size"],
                config["beta"],
                config["gamma"],
                config["temperature_l"],
                config["normalized"],
                epoch,
                optimizer,
            )

        if args.save_model:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    print("Done.")


if __name__ == "__main__":
    main()
