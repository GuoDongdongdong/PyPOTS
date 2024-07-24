import os
import yaml

from utils.experiment import Experiment
from utils.tools import fix_random_seed


def main() :
    conf        = yaml.safe_load(open('conf.yaml', 'r'))
    random_seed = conf['base']['random_seed']
    is_train    = conf['base']['is_train']
    fix_random_seed(random_seed)
    # for pypots enable nni
    os.environ['enable_tuning'] = 'True'
    exp = Experiment(conf)

    if is_train:
        exp.fit()
    else:
        exp.impute()

if __name__ == '__main__' :
    main()