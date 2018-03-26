#!/bin/bash

#arguments:
#$1 - source dir
#$2 - destination dir
#$3 - number of files
echo "calling parallel_stagein.py"
#echo $@
python ../networks/parallel_stagein.py --target ${2} --workers 8 --count ${3} --mkdir ${1}
if [ -f ${1}/stats_train.npz ]; then
    cp ${1}/stats_train.npz ${2}/
fi

