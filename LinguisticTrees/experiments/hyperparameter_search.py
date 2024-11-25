# LinguisticTrees/experiments/hyperparameter_search.py
from ray import tune
import ray
from train_tree_matching import train_model
from COMMON.src.utils.config import cfg

def trainable(config):
    """Wrapper for ray tune"""
    cfg.TRAIN.LEARNING_RATE = config["lr"]
    cfg.TRAIN.BATCH_SIZE = config["batch_size"]
    cfg.MODEL.N_PROP_LAYERS = config["n_layers"]
    
    metrics = train_model(return_metrics=True)
    tune.report(accuracy=metrics["val_accuracy"])

def run_hyperparameter_search():
    ray.init()
    
    analysis = tune.run(
        trainable,
        config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            "n_layers": tune.choice([3, 4, 5, 6])
        },
        num_samples=20,
        resources_per_trial={"cpu": 4, "gpu": 1}
    )
    
    print("Best config:", analysis.best_config)
