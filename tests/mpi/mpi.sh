#!/usr/bin/env bash

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mpiexec --version

mpiexec -n $1 python3 ${DIR}/../../examples/example_dahlquist.py
