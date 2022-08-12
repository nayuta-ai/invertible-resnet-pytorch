import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for experiments')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    args = parser.parse_args()
    return args
