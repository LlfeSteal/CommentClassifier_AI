#!/bin/bash

#OAR -n  TensorflowTester
#OAR -l /nodes=1/gpudevice=1,walltime=00:10:30
#OAR --stdout sortie.out
#OAR --stderr sortie.err
#OAR --project pr-m2-text-mining

source /applis/common/miniconda3/bin/activate fdt
cd ~/projetfouilledetexte/src
python3 tester.py
