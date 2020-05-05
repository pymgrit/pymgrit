#!/usr/bin/env python

import sys
import os
import re

import numpy as np


if __name__ == '__main__':
    conv_regex = re.compile(r'.*conv:\s(\d+\.\d+e-\d+).*')

    # Read all input from stdin
    stdin = sys.stdin.read()

    # Split into lines
    stdin = stdin.split("\n")

    # Remove whitespace
    stdin = list(map(lambda line: line.strip(), stdin))

    # Filter all lines that contain "conv:"
    conv = list(filter(lambda line: 'conv:' in line, stdin))

    # Get the conv values as floats
    convs_actual = []
    for c in conv:
        convs_actual.append(float(conv_regex.match(c).group(1)))

    # Get expected results from results directory
    filename = os.path.dirname(os.path.realpath(__file__)) + '/results/' + sys.argv[1]
    with open(filename) as f:
        convs_desired = list(map(float, f.read().split()))

    # Compare results
    np.testing.assert_almost_equal(convs_actual, convs_desired)

    print('Successfully tested mpi test script with %s tasks' % sys.argv[1])
