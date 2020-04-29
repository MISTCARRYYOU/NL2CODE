import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    # PATHS
    arg_parser.add_argument('--raw_path', default='./dataset/data_conala/', type=str,
                            help='path to the raw target file')
    arg_parser.add_argument('--train_path', default=10, type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--test_path', default=10, type=str,
                            help='path to the eval model file')

    # MODE
    arg_parser.add_argument('--mode', choices=['train', 'test'], required=True,
                            help='Run mode')

    return arg_parser
