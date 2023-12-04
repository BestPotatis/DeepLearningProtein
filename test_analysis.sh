#!/bin/sh
### General options

### -- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J dl_protein_analysis_test

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=32GB]"

### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 33GB

### no gpu needed for testing

# request 32GB of GPU-memory
#BSUB -R "select[gpu32gb]"

### -- set walltime limit: hh:mm --
#BSUB -W 00:10

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s214655@student.dtu.dk

### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o analysis_test_%J.out
#BSUB -e analysis_test_%J.err

# here follow the commands you want to execute with input.in as the input file
module load numpy
module load cuda/11.8

pip3 install scikit-learn --user
pip3 install torch --user

# here follow the commands you want to execute with input.in as the input file
python3 test_analysis.py