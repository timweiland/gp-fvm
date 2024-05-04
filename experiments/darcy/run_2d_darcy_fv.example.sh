#!/bin/bash
#SBATCH --job-name="2d-darcy-fv"
#SBATCH --partition=#TODO
#SBATCH --time 0-20:00:00 # set maximum allowed runtime to 20 hours
#SBATCH --output=#TODO/2d_darcy/fv/logs/slurm/%x-%j.out

# useful for debugging
#scontrol show job $SLURM_JOB_ID
#pwd

eval "$(conda shell.bash hook)"
conda activate #TODO

source #TODO/gp-fvm/experiments/.env

export PRINTFILE=${OUTPUT_DIR}/2d_darcy/fv/logs/slurm/print-$SLURM_JOB_ID.out
export OUTPUTFOLDER=${OUTPUT_DIR}/2d_darcy/fv/results/$SLURM_JOB_ID

mkdir -p $OUTPUTFOLDER

export BETA=1.0
export HDF=${DATA_DIR}/2D_DarcyFlow_beta$BETA.hdf5

export N_BC=30
export N_PDE_XY=$1

echo $HDF

srun python3 -u ${CODE_DIR}/experiments/darcy/run_2d_darcy_fv.py --hdf $HDF --beta $BETA --N-bc $N_BC --N-pde-xy $N_PDE_XY --results-path $OUTPUTFOLDER > $PRINTFILE
