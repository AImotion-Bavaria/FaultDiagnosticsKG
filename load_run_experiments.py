from pathlib import Path
import copy
import yaml

from load_main import main
from utils_eval import summarize_load_results

with open("load_config.yaml", encoding='utf-8') as stream:
    config = yaml.safe_load(stream)

source_domain_ls = ['CWRU_DE', 'CWRU_FE']
seed_ls = [12310, 7151, 333, 391, 1839, 992, 19922, 188, 998, 876]
oc_ls = ['0', '1', '2', '3']
config['path']['model_dir'] = (
    Path.cwd() / 'CaseStudy1_DifferentLoad'
)

models = {
    'CE': ['CrossEntropyLoss', None, None, None],
    'SC': ['SupConLoss', 'CosineSimilarity', 'None', False],
    'SCKG': ['SupConLoss', 'CosineSimilarity', 'None', True],
    'TL': ['TripletMarginLoss', 'LpDistance', 'TripletMarginMiner', False]
}
N = len(source_domain_ls) * len(seed_ls) * len(models.keys()) * len(oc_ls)


def run(configuration, run_i, run_n):
    '''
    Execute load_main.py based on configuration.

    Args:
        configuration: Configuration dictionary
        run_i: Run number
        run_n: Number of total runs
    '''
    configuration_in = copy.deepcopy(configuration)
    name = (
        f'{configuration_in["data"]["source_domain"]}_'
        f'{configuration_in["data"]["oc_target"]}_'
        f'{configuration_in["seed"]}_'
        f'{configuration_in["model"]["loss_fn"]}_'
        f'{configuration_in["model"]["distance"]}_'
        f'{configuration_in["model"]["mining_func"]}_'
        f'{configuration_in["kg"]["kg_trainer"]}'
    )
    print(f'Run {run_i}/{run_n}: {name}')
    main(
        config=configuration_in,
        plots=False,
        save_model=True,
        verbose=False
    )


i = 0
for dom in source_domain_ls:
    config['data']['source_domain'] = dom
    for oc in oc_ls:
        config['data']['oc_target'] = oc
        for seed in seed_ls:
            config['seed'] = seed
            for loss in models.values():
                config['model']['loss_fn'] = loss[0]
                config['model']['distance'] = loss[1]
                config['model']['mining_func'] = loss[2]
                config['kg']['kg_trainer'] = loss[3]
                i += 1
                run(config, i, N)

res_mean, res_std = summarize_load_results(
    Path(config['path']['model_dir']) / 'results_raw.csv'
)
