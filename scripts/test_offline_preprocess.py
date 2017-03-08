#!/usr/bin/env python
DESCRIPTION = """
Explain the script here
"""
import argparse

def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('-o', '--output', help='<PATH> The output folder', type=str, required=True)
    return parser

# Driver function
def main():
    parser = make_arg_parser()
    args = parser.parse_args()

# Used for thread safety
if __name__ == '__main__':
    main()
