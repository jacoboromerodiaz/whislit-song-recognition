from torch import (  # pylint: disable=no-name-in-module
    no_grad,
    cuda,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module

device = "cuda" if cuda.is_available() else "cpu"

def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimiser: Optimizer,
    triplet_loss: Module,
) -> None:
    """Train the model for a single epoch.

    Parameters
    ----------
    model : Module
        Segmentation model.
    train_loader : DataLoader
        DataLoader yielding training mini-batches.
    optimiser : Adam
        The optimiser.
    triplet_loss : Module
        The loss function.

    """
    model.train()
    train_loss = []
    
    for positive_audio, negative_audio, song_embedding in train_loader:
        optimiser.zero_grad()  # clear gradients
        positive_audio = positive_audio.to(device)
        negative_audio = negative_audio.to(device)
        song_embedding = song_embedding.to(device)
        
        positive_embedding, _, _ = model(positive_audio)
        negative_embedding, _, _ = model(negative_audio)

        loss = triplet_loss(song_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimiser.step()
    
        train_loss.append(loss.item())
    return train_loss


@no_grad() 
def validate_epoch(
    model: Module,
    train_loader: DataLoader,
    triplet_loss: Module,
) -> None:
    """Validate the model for a single epoch.

    Parameters
    ----------
    model : Module
        Segmentation model.
    train_loader : DataLoader
        DataLoader yielding training mini-batches.
    triplet_loss : Module
        The loss function.

    """
    model.eval()
    valid_loss =[]

    for positive_audio, negative_audio, song_embedding in train_loader:
        positive_audio = positive_audio.to(device)
        negative_audio = negative_audio.to(device)
        song_embedding = song_embedding.to(device)
        
        positive_embedding, _, _ = model(positive_audio)
        negative_embedding, _, _ = model(negative_audio)

        loss = triplet_loss(song_embedding, positive_embedding, negative_embedding)
        valid_loss.append(loss.item())
        
    return valid_loss
    