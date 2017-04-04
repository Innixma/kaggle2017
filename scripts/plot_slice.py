#!/usr/bin/env python
import argparse
from common import plot_slice, load

DESCRIPTION = """
Explain the script here
"""


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('-s', '--slice', help='<INT> Integer for slice number', type=int, default=80)
    return parser


# Driver function
def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    img = load(args.input)
    plot_slice(img, slice=args.slice)

# Used for thread safety
if __name__ == '__main__':
    main()
