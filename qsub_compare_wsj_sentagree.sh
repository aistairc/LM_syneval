#!/bin/bash
#PBS --group=g-nairobi
#PBS -q gq
#PBS -b 1
#PBS -l cpunum_job=40
#PBS -l gpunum_job=8
#PBS -l elapstim_req=24:00:00
#PBS -v PATH=/system/apps/cuda/9.1.85/bin:${PATH}
#PBS -v LD_LIBRARY_PATH=/system/apps/cudnn/7.1.3/cuda9.1/lib64:/system/apps/cuda/9.1.85/lib64:${LD_LIBRARY_PATH}
#PBS -v CPATH=/system/apps/cudnn/7.1.3/cuda9.1/include:${CPATH}
#PBS -v LIBRARY_PATH=/system/apps/cudnn/7.1.3/cuda9.1/lib64:${LIBRARY_PATH}
#PBS -M h.nouji@gmail.com
#PBS -m e

source $HOME/virtualenvs/torch-0.4.1/bin/activate
cd $PBS_O_WORKDIR

python compare_wsj_sentagree.py -j8
