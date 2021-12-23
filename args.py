import configargparse

def parse(commands = None):
    p = configargparse.ArgParser()
    p.add('-c', '--config', is_config_file = True)
    p.add('--mode', default='test', type=str)
    p.add('--num_cosines', type = int)
    p.add('--dim_subspace', type = int)
    p.add('--dim_diffspace', type = int)
    p.add('--method', type = str)
    p.add('--expname', type=str)
    p.add('--sigma', type=float)
    p.add('--kernel', type=str)

    if commands:
        opt = p.parse_args(commands)
    else:
        opt = p.parse_args()

    return opt