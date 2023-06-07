import os
import logging
import yaml
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

import model
import utils
import utils_data
import utils_kg
import utils_eval
from bearing_transfer import bearing_transfer


def main(
        config=None, plots=True, save_model=True, save_loss=True, verbose=True
    ):
    '''
    Run complete training and evaluation pipeline for bearing
    domain shift.
    1. Load and preprocess data.
    2. Train and test model on source bearing.
    3. Evaluate model on target bearing.
    4. Save results.

    Args:
        config: Configuration dictionary
        plots: Plot evaluation results
        save_model: Save models
        save_loss: Save loss for few shot learning
        verbose: Print information on pipeline run
    '''
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if config is None:
        # Load configuration file
        with open("bearing_config.yaml", encoding='utf-8') as stream:
            config = yaml.safe_load(stream)

    # Get cpu or gpu device for training.
    device = model.get_device()

    # Set random seed
    utils.set_seed(config['seed'])

    # Adjust settings based on config
    config = utils.adjust_config(config)

    # Load source domain data
    data = utils_data.cwru_load(
        path=config['path'][config['data']['source_domain']],
        dom=config['data']['source_domain'],
        sample_rate=config['data']['sample_rate'],
        segment_length=config['data']['segment_length']
    )

    # Load KG embedding (if selected)
    if config['kg']['kg_trainer']:
        g_emb, ent_to_index = utils_kg.get_kg_embedding(
            g_file=config['kg']['file_name']
        )
    else:
        ent_to_index, g_emb = None, None

    # Create class mappings
    config['data']['fault_mapping'] = utils_data.create_mappings(
        config['data']['target'],
        ent_to_index,
        data,
        config['kg']['kg_trainer']
    )

    # Create target columns
    data = utils_data.create_target_columns(
        data,
        config['data']['target'],
        config['data']['fault_mapping']['data_target']
    )

    # Split data into x and y
    x_data, y_data = utils_data.get_x_y(data)

    # Train test split
    config['data']['n_classes'] = len(set(y_data))
    x_train, x_test, \
        y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2,
            random_state=config['seed']
        )
    train_data = [x_train, y_train]
    test_data = [x_test, y_test]

    # Create kg embedding target (if selected)
    if config['kg']['kg_trainer']:
        y_emb = utils_kg.create_y_emb(
            y_train, y_test, config['data']['fault_mapping'], g_emb
        )
        train_data.append(y_emb[0])
        test_data.append(y_emb[1])

    # Create tensor datasets
    train_ds = utils_data.create_tensor_dataset(*train_data)
    test_ds = utils_data.create_tensor_dataset(*test_data)

    # Create data loader for model training
    train_dl_train = DataLoader(
        dataset=train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    test_dl_train = DataLoader(
        dataset=test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    # Create model
    model_net = model.Enc(
        loss_fn=config['model']['loss_fn'],
        n_classes=config['data']['n_classes'],
        emb_size=config['model']['emb_size'],
        projection_size=config['model']['projection_size']
    ).to(device)

    # Configer model
    optimizer, loss_fn, mining_func = model.configer_model(
        model_net,
        loss_fn=config['model']['loss_fn'],
        distance=config['model']['distance'],
        reducer=config['model']['reducer'],
        mining_func=config['model']['mining_func'],
        optimizer=config['training']['optimizer'],
        lr=config['training']['lr']
    )

    # Train model
    epochs = config['training']['epochs']
    config['train_loss'] = []
    config['test_loss'] = []
    for epoch_t in range(epochs):
        logging.info("Epoch %i", epoch_t+1)
        train_loss_e, config['clf_train_score'] = model.train_main(
            train_dl_train, model_net, loss_fn,
            optimizer, device, return_res=True,
            mining_func=mining_func
        )
        config['train_loss'].append(train_loss_e)

        test_loss_e, config['clf_test_score'] = model.test_main(
            test_dl_train, model_net, loss_fn, device,
            return_res=True
        )
        config['test_loss'].append(test_loss_e)
        logging.info("-------------------------------")

    # Save model and configuration
    path, config['name'] = utils_eval.get_save_path(config)
    model.save_model(model_net, path)

    # Get embeddings of trained model
    x_emb_train = model.get_x_emb(train_ds, model_net, device)
    x_emb_test = model.get_x_emb(test_ds, model_net, device)

    # Classify faults based on embedding
    if config['model']['loss_fn'] == "CrossEntropyLoss":
        clf = model_net.lin_fin
    else:
        utils.set_seed(config['seed'])
        config['clf_train_score'], \
            config['clf_test_score'], \
            clf = model.create_linear_model_clf(
                x_emb_train, x_emb_test,
                y_train, y_test
            )
    torch.save(clf, path / 'clf.pt')

    if plots:
        # Plot results (Loss, train & test embedding/acc)
        print("Evaluating model.")
        utils_eval.evaluate_model_bearing(
            x_emb_train, x_emb_test,
            y_train, y_test,
            config, path
        )

    # Save results
    config = utils.prune_config(config)
    utils_eval.save_config(config, path)

    # Evaluate and fine tune on different bearing (FE / DE)
    config = bearing_transfer(
        net=model_net,
        clf=clf,
        config=config,
        path=path,
        seed=config['seed'],
        plots=plots,
        save_loss=save_loss
    )

    # Write results to csv file
    utils_eval.write_to_results(
        config,
        path_results=config['path']['model_dir']
    )

    if not save_model:
        # Remove everything except yaml result files
        if os.path.isfile(path / 'model.pt'):
            os.remove(path / 'model.pt')
        if os.path.isfile(path / 'clf.pt'):
            os.remove(path / 'clf.pt')


if __name__ == '__main__':
    main(plots=True, save_model=True, save_loss=True, verbose=True)
