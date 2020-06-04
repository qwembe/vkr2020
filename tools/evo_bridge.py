import evo.main_traj as traj

def process_traj(*args):
    parser = traj.parser().parse_args(args)
    traj.run(parser)

