"""Train the Light R-CNN student with torchvision Faster R-CNN losses."""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from light_rcnn.data import build_coco_loader
from light_rcnn.model import LightRCNN
from light_rcnn.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coco.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    seed_everything(config["seed"])
    data = config["data"]
    loader = build_coco_loader(data["train_images"], data["train_annotations"],
                               config["train"]["batch_size"], config["train"]["workers"], True)
    model_cfg = config["model"]
    model = LightRCNN(
        num_classes=config["num_classes"],
        bifpn_channels=model_cfg["bifpn_channels"],
        roi_channels=model_cfg["roi_channels"],
        roi_size=model_cfg["roi_size"],
        pretrained_backbone=model_cfg["pretrained"],
        rpn_pre_nms_topk_test=model_cfg["rpn_pre_nms_topk_test"],
        rpn_post_nms_topk_test=model_cfg["rpn_post_nms_topk_test"],
        min_size=config["image_size"],
        max_size=config["image_size"],
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and config["train"]["amp"]))
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(loader.dataset)} images with {device}; checkpoints: {output_dir}")
    for epoch in range(config["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config['train']['epochs']}")
        for images, targets in progress:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.3f}")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch + 1, "config": config}
        torch.save(checkpoint, output_dir / "last.pth")
        print(f"epoch {epoch + 1}: mean loss={running_loss / max(len(loader), 1):.4f}")
    print("Training finished. Add the planned COCO validation/evaluation step before treating a checkpoint as an accepted result.")


if __name__ == "__main__":
    main()
