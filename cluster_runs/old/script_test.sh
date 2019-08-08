#!/bin/bash

#SBATCH --ntasks 512 # Number of processes (Max is < 1088)
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
# #SBATCH --spread-job
# #SBATCH --nodelist=wrcomp[1-4] #
#SBATCH --exclusive
#SBATCH -o machine_test_test.%j.out
#SBATCH -e machine_test_test.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL # Which events should trigger an email
#SBATCH --mail-user=jehahne@uni-wuppertal.de # Mail address for notifications, also check your spam folder


date
mpirun --mca btl tcp,sm,self python3 -m mpi4py -m cluster_runs.run_machine_test
date
