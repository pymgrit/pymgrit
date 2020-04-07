#!/usr/bin/env bash

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

mpiexec --version

mpiexec -n $1 --use-hwthread-cpus --oversubscribe python3 ${DIR}/../../examples/example_dahlquist.py
