import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--raw_path', default='./dataset/data_conala/', type=str,
                            help='path to the raw target file')
    arg_parser.add_argument('--train_path', default=10, type=str,
                            help='path to the training target file')
    arg_parser.add_argument('--test_path', default=10, type=str,
                            help='path to the eval model file')

    arg_parser.add_argument('--mode', type=str,
                            help='path to the raw target file')

    return arg_parser
