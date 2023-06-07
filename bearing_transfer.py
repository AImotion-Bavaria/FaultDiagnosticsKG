import copy
import string
from pathlib import Path
import logging

import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils
import utils_data
import utils_eval
import model


def bearing_transfer(
        net, clf, config, path, seed, plots=True, save_loss=True
        ):
    '''
    Transfer trained model to new bearing.

    Args:
        net: Encoder network
        clf: Classifier
        config: Configuration dictionary
        path: Path to trained model
        seed: Random seed
        plots: Plot results
        save_loss: Save loss to results

    Returns:
        config: Configuration dictionary with results
    '''
    device = model.get_device()

    if config['data']['source_domain'] == 'CWRU_FE':
        dom = 'CWRU_DE'
    else:
        dom = 'CWRU_FE'

    # Load classifiers
    if config['model']['loss_fn'] == "CrossEntropyLoss":
        sub_folder = 'DirectClassification'
        if clf is None:
            clf = net.lin_fin
    else:
        sub_folder = 'Embedding'
        if clf is None:
            clf = torch.load(path / 'clf.pt')

    # Create paths for results
    m_dir = Path(config['path']['model_dir'])
    m_folder = config['name']
    path_out_model = Path(
        m_dir /
        sub_folder /
        m_folder
    )

    # Create fault mappings
    class_label_mapping = config['data']['fault_mapping']

    # Create Transfer results yaml
    try:
        config_transfer = config['transfer']['results']
    except KeyError:
        config_transfer = {}

    # Settings for few shot transfer learning
    shots = config['transfer']['shots']
    epochs = config['transfer']['epochs']

    logging.info("Evaluating on %s.", dom)
    utils.set_seed(seed)
    df_dom = utils_data.cwru_load(
            path=Path(config['path'][dom]),
            dom=dom,
            sample_rate=config['data']['sample_rate'],
            segment_length=config['data']['segment_length'],
        )

    if dom not in config_transfer:
        res_dom = {}
    else:
        res_dom = config_transfer[dom]

    # Create output path
    path_out = Path(
        path_out_model /
        f"{config['data']['source_domain']}_to_{dom}"
    )

    # Create target columns
    df_dom = utils_data.create_target_columns(
            df_dom,
            config['data']['target'],
            config['data']['fault_mapping']['data_target']
        )
    # Get features x and target y
    x, y = utils_data.get_x_y(df_dom)

    # Get embeddings
    x_emb = model.get_x_emb(
        utils_data.create_tensor_dataset(x, y),
        net,
        device
    )

    # Calculate accuracy (zero shot)
    res_dom['test_acc'] = utils_eval.get_accuracy(
            clf, x_emb, y, device
        )

    # Fine tune classifier on domain
    for random_state in config['transfer']['seeds']:
        res_dom_seed = {}
        for n_shots in shots:
            res_dom_seed[n_shots] = {}
            res_dom_seed[n_shots] = n_shot_learning(
                clf, n_shots, x_emb, y, epochs,
                random_state, device,
                save_loss=save_loss
            )
        res_dom[str(random_state)] = res_dom_seed
    config_transfer[dom] = res_dom

    if plots:
        # Create output directory
        path_out.mkdir(exist_ok=True)
        create_domain_transfer_plots(
            config_transfer[dom],
            class_label_mapping['data_target'],
            x_emb,
            y,
            shots,
            path_out,
            seed=config['transfer']['seeds'][-1]
        )

    config['transfer']['results'] = config_transfer

    with open(path_out_model / 'config_loss.yaml',
              'w',
              encoding='utf-8') as out:
        yaml.dump(config, out)

    return config


def n_shot_learning(net_clf, n_shot, data, target,
                    epochs, random_state, device,
                    save_loss=True):
    '''
    Fine tune final layer or seperate classifier on n shots / samples.

    Args:
        net_clf: Encoder network
        n_shot: List of number of shots for fine tuning
        data: Target domain data
        target: Target column
        epochs: Number of epochs for fine tuning
        random_state: List of random seeds for fine sampling for
            few shot learning
        device: Device on which model is trained
        save_loss: Save loss

    Return:
        res: Results for few shot learning
    '''
    # Create deep copy of network
    net_clf_cp = copy.deepcopy(net_clf)

    utils.set_seed(random_state)

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, train_size=n_shot*len(set(target)),
        stratify=target, random_state=random_state
    )
    train_dl = utils_data.create_dataloader(x_train, y_train)
    test_dl = utils_data.create_dataloader(x_test, y_test)

    # Configer model
    loss_fn, optimizer = model.model_standard_config(net_clf_cp)

    res = {}
    res['train_loss'] = []
    res['test_loss'] = []
    for epoch_t in range(epochs):
        logging.info("Epoch %i", epoch_t+1)
        res['train_loss'].append(
            model.train_lin_mod(
                train_dl, net_clf_cp, loss_fn, optimizer,
                device, return_res=True
            )[0]
        )
        res['test_loss'].append(
            model.test_lin_mod(
                test_dl, net_clf_cp, loss_fn, device, return_res=True
            )[0]
        )
        logging.info("-------------------------------")

    if not save_loss:
        res.pop('train_loss')
        res.pop('test_loss')

    res['train_acc'] = utils_eval.get_accuracy(
        net_clf_cp, x_train, y_train, device
    )
    res['test_acc'] = utils_eval.get_accuracy(
        net_clf_cp, x_test, y_test, device
    )

    return res


def create_domain_transfer_plots(
        config_transfer, fault_mapping, x_emb, y, shots, path, seed
        ):
    '''
    Create plot for domain transfer evaluation.

    Args:
        config_transfer: Configuration dictionary
        fault_mapping: Mapping of numbers to labels
        x_emb: Feature embedding
        y: Labels
        shots: Number of shots used for fine tuning
        path: Path for saving plots
        seed: Random seed for title
    '''
    test_score_0 = f"(Mean Clf-Accuracy: {config_transfer['test_acc']:.3f})"
    config_transfer = config_transfer[str(seed)]

    # Get 2d embedding
    x_emb_2d = utils_eval.get_2d_embedding(
        x_emb.detach().cpu().numpy(), perp=50
    )

    # Setup figure
    layout = [['A', string.ascii_lowercase[i]]
              for i, _ in enumerate(shots)]
    fig, axes = plt.subplot_mosaic(layout, figsize=(15, 8))
    fig.suptitle(f'Results for sampling with seed: {seed}')
    if len(layout) > 1:
        for letter in layout[:-1]:
            axes['a'].get_shared_x_axes().join(axes[letter[1]])
            axes[letter[1]].set_xticklabels([])

    # Plot zero shot clustering
    axes['A'].set_title(f'T-SNE 2d embedding {test_score_0}')
    utils_eval.plot_2d_embedding(
        x_emb_2d,
        y,
        mapping=fault_mapping,
        ax=axes['A']
    )

    # Plot loss and accuracy for few shot learning
    for i, letter in enumerate(layout):
        ax = axes[letter[1]]
        n_shots = shots[i]
        train_score = config_transfer[n_shots]['train_acc']
        train_score = f"Train score: {train_score:.3f}"
        test_score = config_transfer[n_shots]['test_acc']
        test_score = f"Test score: {test_score:.3f}"
        utils_eval.plot_loss(config_transfer[n_shots], ax=ax)
        ax.set_title(f'{n_shots}-Shots | Loss | {train_score} {test_score}')

    # Save plot
    fig.savefig(path / 'results_transfer.png')
