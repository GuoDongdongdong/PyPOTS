from datetime import datetime
import os
import yaml

from utils.experiment import Experiment
from utils.tools import fix_random_seed, set_logger

def parse_conf():
    conf = yaml.safe_load(open('conf.yaml', 'r'))
    return conf

def init():
    # for pypots enable nni
    os.environ['enable_tuning'] = 'True'
    conf = parse_conf()
    # fix random seed
    random_seed = conf['base']['random_seed']
    fix_random_seed(random_seed)
    # set logger
    time_now = datetime.now().__format__("%Y%m%d_T%H%M%S")
    saving_dir = conf['base']['saving_dir']
    log_file_name = conf['base']['log_file_name']
    set_logger(saving_dir=os.path.join(saving_dir, time_now), file_name=log_file_name)

    return conf

def main() :
    conf = init()
    exp = Experiment(conf)
    is_train = conf['base']['is_train']
    if is_train:
        exp.fit()
    else:
        exp.impute()

if __name__ == '__main__' :
    main()