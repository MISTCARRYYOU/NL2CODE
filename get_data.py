from common.utils import init_arg_parser
from dataset.preprocess_dataset import preprocess_data
from dataset.json_to_seq2seq import data_creation


def init_config(arg_parser):
    args = arg_parser.parse_args()

    return args

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config(arg_parser)
    preprocess_data(args)
    data_creation(args)
