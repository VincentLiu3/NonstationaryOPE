import argparse


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_parser():
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--env_type', type=str, default='', help='yeast or movie')
    parser.add_argument('--window_size', type=int, default=1, help='window size')
    parser.add_argument('--num_round', type=int, default=50, help='number of round')
    parser.add_argument('--num_run', type=int, default=30, help='number of runs')
    parser.add_argument('--n', type=int, default=1000, help='sample size for each round')
    parser.add_argument('--sample_percentage', type=float, default=0.5, help='sample percentage for each round')
    parser.add_argument('--generate_data', type=str2bool, default=False, help='re-generate data or not')
    parser.add_argument('--sample_data', type=str2bool, default=False, help='re-sample data or not')
    return parser
