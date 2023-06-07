import copy
import yaml
from pathlib import Path

from bearing_main import main
from utils_eval import summarize_bearing_results

with open("bearing_config.yaml", encoding='utf-8') as stream:
    config = yaml.safe_load(stream)

source_domain_ls = ['CWRU_DE', 'CWRU_FE']
config['path']['model_dir'] = (
    Path.cwd() / 'CaseStudy2_DifferentBearing'
)
seed_ls = [12310, 7151, 333, 391, 1839, 992, 19922, 188, 998, 876]
config['transfer']['seeds'] = [
    2482,  139, 7931, 5018, 9051, 8872, 8608,  468, 8864, 4837
]

models = {
    'CE': ['CrossEntropyLoss', None, None, None],
    'SC': ['SupConLoss', 'CosineSimilarity', 'None', False],
    'SCKG': ['SupConLoss', 'CosineSimilarity', 'None', True],
    'TL': ['TripletMarginLoss', 'LpDistance', 'TripletMarginMiner', False]
}
N = len(source_domain_ls) * len(seed_ls) * len(models.keys())


def run(configuration, run_i, run_n):
    '''
    Execute bearing_main.py based on configuration.

    Args:
        configuration: Configuration dictionary
        run_i: Run number
        run_n: Number of total runs
    '''
    configuration_in = copy.deepcopy(configuration)
    name = (
        f'{configuration_in["data"]["source_domain"]}_'
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
        save_loss=False,
        verbose=False
    )


i = 0
for dom in source_domain_ls:
    config['data']['source_domain'] = dom
    for seed in seed_ls:
        config['seed'] = seed
        for loss in models.values():
            config['model']['loss_fn'] = loss[0]
            config['model']['distance'] = loss[1]
            config['model']['mining_func'] = loss[2]
            config['kg']['kg_trainer'] = loss[3]
            i += 1
            run(config, i, N)

res_mean, res_std = summarize_bearing_results(
    Path(config['path']['model_dir']) / 'results_raw.csv',
    config['transfer']['shots']
)
