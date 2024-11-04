import torch

from train_utils import parse_command_line
from model import SoundNet, TripletLoss

import yaml  # type: ignore
import tqdm  # type: ignore
from pathlib import Path

from dataset import get_dataloaders
from engine import train_epoch, validate_epoch

def train_soundnet(config: dict):
    train_loader, valid_loader = get_dataloaders(config)
    
    print(f"Dataset size: train: {len(train_loader)}, valid: {len(valid_loader)}")
    
    model = SoundNet()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    triplet_loss = TripletLoss(margin=1.0)
    
    for _, epoch in tqdm.tqdm(
        enumerate(range(config.get("epochs", 0))),
        ascii=False,  
        total=config.get("epochs", 0),
    ):
        train_epoch(
            model,
            train_loader,
            optimizer,
            triplet_loss,
        )
        
        if epoch % 2 == 0:
                validate_epoch(
                model,
                train_loader,
                optimizer,
                triplet_loss,
            )
            
    save_model_path = Path(config.get("save_dir", "./")) / "checkpoints"
    save_model_path.mkdir(exist_ok=True)

    torch.save(model.state_dict(), save_model_path / f"weights.pth")

if __name__ == "__main__":
    args = parse_command_line()
    with open(args.config, encoding="utf-8") as f:
        config_file = yaml.safe_load(f)
    train_soundnet(config_file)