import os
import sys
import argparse
import logging
import lc

def init_logging(exp_dir, filename='exp.log', loglevel = logging.DEBUG):

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Initialize logging LOGLEVEL to file
    logging.basicConfig(filename = os.path.join(exp_dir, filename),
                        level = loglevel,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]')

    logger = logging.getLogger()

    # Initialize logging INFO to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def parse_commandline_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--action', type=str, default='train')
    arg_parser.add_argument('--data-file', type=str, default='data_train.csv')
    arg_parser.add_argument('--exp-dir', type=str, default='exp/linear')
    arg_parser.add_argument('--param-file', type=str, default='params.linear')
    
    return arg_parser.parse_args()


def run():
    args = parse_commandline_args()
    init_logging(args.exp_dir)
    logging.info("%s", " ".join(sys.argv))

    logging.info('---------- ')
    logging.info('Started program with action: %s', args.action)
    logging.info('---------- ')

    if args.action == 'prepare':
        lc.prepare_data()
    elif args.action == 'eda':
        lc.eda(args.data_file, args.param_file, args.exp_dir)
    elif args.action == 'train':
        lc.train(args.data_file, args.param_file, args.exp_dir)
    elif args.action == 'test':
        lc.predict(args.data_file, args.exp_dir, scoring = True)
    elif args.action == 'predict':
        lc.predict(args.data_file, args.exp_dir, scoring = False)
    else:
        raise RuntimeError('Action %s not supported' % args.action)


if __name__ == '__main__':
    run()
