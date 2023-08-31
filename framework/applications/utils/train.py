
import logging
import torch
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def freeze_batch_norm_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()


def train_classification_model(model, optimizer, criterion, trainloader, device,
                         verbose=True, max_batches=None, freeze_batch_norm=False):
    
    """
    Parameters
    ----------
    device: torch.device
        Choose between cuda or cpu.
    model: torch.nn.Module
        A pytorch network model.
    optimizer: torch.optim.Optimizer
        A pytorch optimizer like Adam.
    criterion: torch.nn.Loss
        A pytorch criterion that defines the loss.
    trainloader: torch.utils.data.DataLoader
        Loader of train data.
    max_batches: int
        How many batches the model should train for.
    verbose: bool
        If True, print text - verbose mode.
    freeze_batch_norm: bool
        If True set batch norm layers to eval. Default: False

    Returns
    -------
    success: bool
        Returns False is nans encountered in the loss else True.
    """
    model.to(device)
    model.train()
    if freeze_batch_norm:
        freeze_batch_norm_layers(model)

    train_loss = []
    correct = 0
    total = 0

    total_iterations = max_batches or len(trainloader)
    iterator = tqdm(enumerate(trainloader), total=total_iterations, position=0, leave=True, desc='train_classification') \
        if verbose else enumerate(trainloader)

    for batch_idx, (inputs, targets) in iterator:

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            LOGGER.warning('--> Loss is Nan.')
            break

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx == max_batches:
            break

    acc = correct * 100.0 / total
    mean_train_loss = np.mean(train_loss)
    
    return acc, mean_train_loss
##################################################################