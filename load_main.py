import logging
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import model
import utils
import utils_data
import utils_kg
import utils_eval


def main(config=None, plots=True, save_model=True, verbose=True):
    '''
    Run complete training and evaluation pipeline for operating
    condition domain shift.
    1. Load and preprocess data.
    2. Train and test model on source operating condition.
    3. Evaluate model on target operating condition.
    4. Save results.

    Args:
        config: Configuration dictionary
        plots: Plot evaluation results
        save_model: Save models
        verbose: Print information on pipeline run
    '''
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if config is None:
        # Load configuration file
        with open(
            "load_config.yaml", encoding='utf-8'
        ) as stream:
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

    # Split data according to operating conditions
    oc_target = str(config['data']['oc_target'])
    data_oc_source, data_oc_target = utils_data.split_oc(data, oc_target)

    # Split data into x and y
    x_data, y_data = utils_data.get_x_y(data_oc_source)
    x_data_oc, y_data_oc = utils_data.get_x_y(data_oc_target)

    # Train test split
    config['data']['n_classes'] = len(set(y_data))
    x_train, x_test, \
        y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2,
            random_state=config['seed']
        )
    train_data = [x_train, y_train]
    test_data = [x_test, y_test]
    oc_data = [x_data_oc, y_data_oc]

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
    oc_target_ds = utils_data.create_tensor_dataset(*oc_data)

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
            test_dl_train, model_net, loss_fn, device, return_res=True
        )
        config['test_loss'].append(test_loss_e)
        logging.info("-------------------------------")

    # Save model and configuration
    path, config['name'] = utils_eval.get_save_path(config)
    model.save_model(model_net, path)

    # Get embeddings with trained model
    x_emb_train = model.get_x_emb(train_ds, model_net, device)
    x_emb_test = model.get_x_emb(test_ds, model_net, device)
    x_emb_oc_target = model.get_x_emb(oc_target_ds, model_net, device)

    # Classify source operating condition faults based on embedding
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

    # Calculate accuracy on target operating conditions
    config['oc_acc'] = utils_eval.get_accuracy(
        clf, x_emb_oc_target, y_data_oc,
        device=device, return_y_pred=False
    )

    if plots:
        # Plot results (Loss, train & test embedding/acc)
        logging.info("Evaluating model.")
        utils_eval.evaluate_model_oc(
            x_emb_test, x_emb_oc_target,
            y_test, y_data_oc,
            config, path
        )

    # Save results
    config = utils.prune_config(config)
    utils_eval.save_config(config, path)

    # Write results to csv file
    utils_eval.write_to_results(
        config,
        path_results=config['path']['model_dir']
    )

    if not save_model:
        # Remove everything except yaml result files
        if (path / 'model.pt').is_file():
            (path / 'model.pt').unlink()
        if (path / 'clf.pt').is_file():
            (path / 'clf.pt').unlink()


if __name__ == '__main__':
    main(plots=True, save_model=True, verbose=True)
