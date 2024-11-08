import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import PlannerLoss, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    # log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    # logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)


    # create loss function and optimizer
    loss_func = PlannerLoss()
    # optimizer = ...
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    val_plannermetric=PlannerMetric()
    # training loop
    for epoch in range(num_epoch):
        
        val_plannermetric.reset()

        model.train()
        
        for batch in train_data:
            track_left, track_right, waypoints,waypoints_mask = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device),batch['waypoints_mask'].to(device)

            
            # TODO: implement training step
            pred = model(track_left,track_right)  
            loss = loss_func(pred,waypoints,waypoints_mask)
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()


        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left, track_right, waypoints,waypoints_mask = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device),batch['waypoints_mask'].to(device)

                # TODO: compute validation accuracy
                pred = model(track_left,track_right)

                val_plannermetric.add(pred,waypoints,waypoints_mask)
               
                

            # "l1_error": float(l1_error),
            # "longitudinal_error": float(longitudinal_error),
            # "lateral_error": float(lateral_error),
            # "num_samples": self.total,
        val_planner = val_plannermetric.compute()
        l1_error =val_planner.l1_error
        longitudinal_error = val_planner.longitudinal_error     
        lateral_error = val_planner.lateral_error
        num_samples = val_planner.num_samples

        

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                # f"train_acc={epoch_train_acc:.4f} "
                f"l1_error={l1_error:.4f}"
                # f" train_abs_depth_error={train_abs_depth_error:.4f} "
                f"longitudinal_error={longitudinal_error:.4f}"
                # f" train_tp_depth_error={train_tp_depth_error:.4f} "
                f"lateral_error={lateral_error:.4f}"
                # f" train_iou={train_iou:.4f} "
                f"num_samples={num_samples:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    # torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    # print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
