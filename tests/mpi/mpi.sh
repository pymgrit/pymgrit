#!/usr/bin/env bash

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mpiexec --version

mpiexec --use-hwthread-cpus -np $1 ${DIR}/../../venv/bin/python3 ${DIR}/../../examples/example_dahlquist.py
