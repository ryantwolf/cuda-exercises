#!/usr/bin/zsh
rm /home/ryan/Repos/ECE408/local/libwb/template.cu || true
cp ./template.cu /home/ryan/Repos/ECE408/local/libwb/template.cu
# Store the current directory in a variable
# so we can return to it later
CURR_DIR=$(pwd)
cd /home/ryan/Repos/ECE408/local/libwb
nvcc -std=c++11 -rdc=true -c template.cu -o template.o
nvcc -std=c++11 -o local_mp template.o /home/ryan/Repos/ECE408/local/libwb/lib/libwb.so
cd $CURR_DIR
mkdir -p local_tests
# Loop over all data files
for i in {0..9};
do
    echo "--------------";
    echo "Dataset " $i 
    /home/ryan/Repos/ECE408/local/libwb/local_mp -e ./data/${i}/output.raw -i ./data/${i}/input0.raw,./data/${i}/input1.raw -o local_tests/${i}.raw -t matrix
done