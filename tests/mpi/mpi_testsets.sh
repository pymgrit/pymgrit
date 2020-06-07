#!/usr/bin/env bash

set -e

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 || (echo "Could not change into current directory!" && exit 1) ; pwd -P )"

mpiexec --quiet -n 1 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 1
mpiexec --quiet -n 2 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 2
mpiexec --quiet -n 3 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 3
mpiexec --quiet -n 4 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 4
mpiexec --quiet -n 5 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 5
mpiexec --quiet -n 6 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 6
mpiexec --quiet -n 7 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/$1.py | ${DIR}/mpi.py $1 7
