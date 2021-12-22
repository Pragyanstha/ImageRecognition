import configargparse

def parse():
    p = configargparse.ArgParser()
    p.add('-c', '--config', is_config_file = True)
    p.add('--num_cosines', type = int)
    p.add('--dim_subspace', type = int)
    p.add('--method', type = str)
    p.add('--expname', type=str)
    p.add('--sigma', type=float)
    p.add('--kernel', type=str)
    return p.parse_args()