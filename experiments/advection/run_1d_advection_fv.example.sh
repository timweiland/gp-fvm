#!/bin/bash
#SBATCH --job-name="1d-advection-fvm"
#SBATCH --partition=#TODO
#SBATCH --time 0-20:00:00 # set maximum allowed runtime to 20 hours
#SBATCH --output=#TODO/output/1d_advection/fv/logs/slurm/%x-%j.out

# useful for debugging
#scontrol show job $SLURM_JOB_ID
#pwd

eval "$(conda shell.bash hook)"
conda activate #TODO

source #TODO/gp-fvm/experiments/.env

export PRINTFILE=${OUTPUT_DIR}/1d_advection/fv/logs/slurm/print-$SLURM_JOB_ID.out
export OUTPUTFOLDER=${OUTPUT_DIR}/1d_advection/fv/results/$SLURM_JOB_ID

mkdir -p $OUTPUTFOLDER

export BETA=0.4
export HDF=${DATA_DIR}/1D_Advection_Sols_beta$BETA.hdf5

export N_PDE_T=$1
export N_PDE_X=$2

echo $HDF

srun python3 -u ${CODE_DIR}/experiments/advection/run_1d_advection_fv.py --hdf $HDF --beta $BETA --N-pde-t $N_PDE_T --N-pde-x $N_PDE_X --results-path $OUTPUTFOLDER > $PRINTFILE
