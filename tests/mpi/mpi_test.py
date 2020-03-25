import nose
import subprocess
import pathlib
import logging
import re

import numpy as np


def _mpi_base_test(num_processes):
    # Get base dir of the project
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

    # Run mpi example
    process = subprocess.Popen(
        ['mpiexec', '-n', str(num_processes), './venv/bin/python3', './examples/example_dahlquist.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=BASE_DIR)

    # Get stdout and stderr output
    stdout, stderr = process.communicate()

    assert stderr == b''

    # Convert output to array
    stdout = str(stdout).split("\\n")

    # Trim lines
    stdout = list(map(lambda x: x.strip(), stdout))

    # Filter lines so only "conv:" lines remain
    stdout = list(filter(lambda x: "conv:" in x, stdout))

    # Get the conv value of every line
    stdout = re.findall(r'.*conv: (\d+\.\d+e-\d+).*', '\n'.join(stdout), re.MULTILINE)

    # Convert to floats
    stdout = list(map(lambda x: float(x), stdout))

    desired = [7.186185937025429e-05, 1.2461067075859538e-06, 2.1015566149418612e-08, 3.144127389557912e-10,
               3.975216519949153e-12]

    np.testing.assert_equal(stdout, desired)


def test_mpi_2n():
    _mpi_base_test(2)


def test_mpi_4n():
    _mpi_base_test(4)


if __name__ == '__main__':
    nose.run()
