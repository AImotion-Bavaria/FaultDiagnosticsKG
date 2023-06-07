import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, reducers, distances

import utils_data


class Enc(nn.Module):
    '''
    Encoder network. Includes Proj() for contrastive losses.
    '''
    def __init__(self, loss_fn, n_classes, emb_size, projection_size):
        super().__init__()

        self.loss_fn = loss_fn
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.projection_size = projection_size

        # Encoder layers (convolutional)
        self.en_conv = [
            nn.Conv1d(1, 64, 12),
            nn.Conv1d(64, 32, 12),
            nn.Conv1d(32, 16, 12),
            nn.Conv1d(16, 8, 12)
        ]

        rep = []
        for conv in self.en_conv:
            rep.append(conv)
            rep.append(nn.LeakyReLU(0.5))
            rep.append(nn.MaxPool1d(kernel_size=12, stride=2))
            rep.append(nn.Dropout1d(p=0.1))

        # Embedding layer
        rep.append(nn.Flatten())
        rep.append(nn.Linear(96, self.projection_size))
        rep.append(nn.LeakyReLU(0.5))

        # Create the encoder
        self.create_rep_space = nn.Sequential(*rep)

        if self.loss_fn == "CrossEntropyLoss":
            self.lin_fin = nn.Linear(self.projection_size, self.n_classes)
        else:
            self.lin_fin = nn.Linear(self.projection_size, self.emb_size)

    def forward(self, x):
        '''
        Forward pass.
        '''
        rep_space = self.create_rep_space(x)
        if self.loss_fn == "CrossEntropyLoss":
            return self.lin_fin(rep_space), rep_space
        return F.normalize(self.lin_fin(rep_space), p=2), rep_space


class LinearModel(nn.Module):
    '''
    Linear model used on embedding learned via contrastive loss.
    '''
    def __init__(self, emb_size, n_classes) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.lin1 = nn.Linear(
            self.emb_size, self.n_classes
        )

    def forward(self, x):
        '''
        Forward pass.
        '''
        return self.lin1(x)


def train_main(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    return_res=False,
    mining_func=None
):
    '''
    Train routine for encoder network.

    Args:
        dataloader: DataLoader
        model: Encoder model
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to be trained on
        return_res: Return train results
        mining_func: Mining function for contrastive loss

    Returns:
        train_loss: Train loss
        correct: Train accuracy
    '''
    model.train()
    train_loss, correct = 0, 0
    for data in dataloader:
        y = data[1].to(torch.long).squeeze().to(device)
        X = data[0].to(torch.float32).to(device)

        pred = model(X)[0].squeeze()

        if mining_func is not None:
            indices_tuple = mining_func(pred, y)

        try:
            y_emb = data[2].to(torch.float32).to(device)
            if isinstance(
                loss_fn, (losses.SupConLoss, losses.TripletMarginLoss)
                    ):
                if mining_func is not None:
                    loss = loss_fn(pred, y, indices_tuple, ref_emb=y_emb)
                else:
                    loss = loss_fn(pred, y, ref_emb=y_emb)
            elif isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(pred, y_emb)
        except IndexError:
            if mining_func is not None:
                loss = loss_fn(pred, y, indices_tuple)
            else:
                loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= len(dataloader)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        correct /= len(dataloader.dataset)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        logging.info("Train loss: %.4f | Train acc: %.4f", train_loss, correct)
    elif isinstance(
        loss_fn, (losses.SupConLoss, losses.TripletMarginLoss, nn.MSELoss)
            ):
        logging.info("Train loss: %.4f", train_loss)

    if return_res:
        return train_loss, float(correct)
    return None


def train_lin_mod(dataloader, model, loss_fn, optimizer,
                  device, return_res=False):
    '''
    Train routine for linear model.

    Args:
        dataloader: DataLoader
        model: Encoder model
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to be trained on
        return_res: Return train results

    Returns:
        train_loss: Train loss
        correct: Train accuracy
    '''
    model.train()
    train_loss, correct = 0, 0
    for x, y in dataloader:
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.long)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    logging.info("Train loss: %.4f | Train acc: %.4f", train_loss, correct)
    if return_res:
        return train_loss, float(correct)
    return None


def test_main(dataloader, model, loss_fn, device, return_res=False):
    '''
    Test routine for encoder network.

    Args:
        dataloader: DataLoader
        model: Encoder model
        loss_fn: Loss function
        device: Device to be trained on
        return_res: Return test results

    Returns:
        test_loss: Train loss
        correct: Train accuracy
    '''
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data[0].to(device), data[1].to(device).squeeze()
            X = X.to(torch.float32)
            y = y.to(torch.long).reshape(-1)
            pred = model(X)[0].squeeze().reshape(len(y), -1)

            try:
                y_emb = data[2].to(torch.float32).to(device)
                if isinstance(
                    loss_fn, (losses.SupConLoss, losses.TripletMarginLoss)
                        ):
                    test_loss += loss_fn(pred, y, ref_emb=y_emb).item()
                elif isinstance(loss_fn, nn.MSELoss):
                    test_loss += loss_fn(pred, y_emb).item()
            except IndexError:
                test_loss += loss_fn(pred, y).item()

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        correct /= len(dataloader.dataset)

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        logging.info("Test loss: %.4f | Test acc: %.4f", test_loss, correct)
    elif isinstance(
        loss_fn, (losses.SupConLoss, losses.TripletMarginLoss, nn.MSELoss)
            ):
        logging.info("Test loss: %.4f", test_loss)

    if return_res:
        return test_loss, float(correct)

    return None


def test_lin_mod(dataloader, model, loss_fn, device, return_res=False):
    '''
    Test routine for linear model.

    Args:
        dataloader: DataLoader
        model: Encoder model
        loss_fn: Loss function
        device: Device to be trained on
        return_res: Return test results

    Returns:
        test_loss: Train loss
        correct: Train accuracy
    '''
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.long)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    logging.info("Test loss: %.4f | Test acc: %.4f", test_loss, correct)
    if return_res:
        return test_loss, float(correct)
    return None


def get_x_emb(dataset, model, device):
    '''
    Get embedding for input x.

    Args:
        dataset: TensorDataset
        model: Model for creating embedding
        device: Device to run model on

    Returns:
        Embedding of input x
    '''
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False
    )
    model.eval()
    x_emb = []
    with torch.no_grad():
        for data in dataloader:
            x_batch = data[0].to(torch.float32).to(device)
            x_res = model(x_batch)[1].detach().cpu()
            if len(x_res.shape) == 1:
                x_res = x_res.unsqueeze(dim=0)
            x_emb.append(x_res)
    return torch.cat(x_emb, dim=0)


def configer_model(m, loss_fn, distance, reducer, mining_func, optimizer, lr):
    '''
    Configer model.

    Args:
        m: Encoder model
        loss_fn: Loss function
        distance: Distnace metric
        device: Device to be trained on
        reducer: Reducer for combining results
        mining_func: Mining function for contrastive loss
        optimizer: Optimizer
        lr: Learning rate

    Returns:
        optimizer: Optimizer
        loss_fn: Loss function
        mining_fn: Mining function
    '''
    if loss_fn in ['SupConLoss', 'TripletMarginLoss']:
        # Configer model with metric loss function

        # Set distance metric
        if distance == 'CosineSimilarity':
            distance = distances.CosineSimilarity()
        elif distance == 'LpDistance':
            distance = distances.LpDistance(
                normalize_embeddings=True, p=2, power=1
            )

        # Set reducer
        if reducer == 'ThresholdReducer':
            reducer = reducers.ThresholdReducer(low=0)
            # Equivalent to mean and AvgNonZeroReducer

        # Set mining function
        if mining_func == 'None':
            mining_func = None
        elif mining_func == 'TripletMarginMiner':
            mining_func = miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="semihard"
            )

        # Set loss function
        if loss_fn == 'SupConLoss':
            loss_fn = losses.SupConLoss(
                distance=distance, reducer=reducer
            )
        elif loss_fn == 'TripletMarginLoss':
            loss_fn = losses.TripletMarginLoss(
                distance=distance, reducer=reducer
            )

    else:
        # Configer model with other loss fucntion

        # Set loss function
        if loss_fn == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss()

        # Set mining function
        mining_func = None

    # Set optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            m.parameters(), lr=lr
        )

    return optimizer, loss_fn, mining_func


def get_device():
    '''
    Check and get GPU if available.
    '''
    device = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info("Using %s device.", device)
    return device


def save_model(model, path):
    '''
    Save model to path.
    '''
    torch.save(model, path / 'model.pt')


def model_standard_config(net):
    '''
    Configer standard model (used for linear model).
    '''
    # Configer model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    return loss_fn, optimizer


def create_linear_model_clf(x_train, x_test, y_train, y_test):
    '''
    Create and evaluate linear model classifier.

    Args:
        x_train: Train features
        x_test: Test features
        y_train: Train labels
        y_test: Test labels

    Returns:
        train_score: Score on training data
        test_score: Score on test data
        lin_mod: Trained linear model
    '''
    emb_size = x_train.shape[1]
    n_classes = len(set(y_train))
    device = get_device()

    # Create dataloaders
    train_dl = utils_data.create_dataloader(x_train, y_train)
    test_dl = utils_data.create_dataloader(x_test, y_test)

    # Create linear model
    lin_mod = LinearModel(emb_size, n_classes).to(device)

    # Configer model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lin_mod.parameters())

    # Train model
    epochs = 20
    for epoch_t in range(epochs):
        logging.info("Epoch %i", epoch_t+1)
        train_score = train_lin_mod(
            train_dl, lin_mod, loss_fn, optimizer, device, return_res=True
        )[1]
        test_score = test_lin_mod(
            test_dl, lin_mod, loss_fn, device, return_res=True,
        )[1]
        logging.info("-------------------------------")

    return train_score, test_score, lin_mod
