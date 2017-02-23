import argparse
import lc


def parse_commandline_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--action', type=str, default='train')
    arg_parser.add_argument('--data-file', type=str, default='data_train.csv')
    arg_parser.add_argument('--exp-dir', type=str, default='exp/linear')
    arg_parser.add_argument('--param-file', type=str, default='params.linear')
    
    return arg_parser.parse_args()


def run():
    args = parse_commandline_args()

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
