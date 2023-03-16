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
mkdir -p bench
# Loop over all data files
for i in {0..5};
do
    echo "--------------";
    echo "Dataset " $i 
    /home/ryan/Repos/ECE408/local/libwb/local_mp -e ./data/${i}/output.dat -i ./data/${i}/input.dat,./data/${i}/kernel.dat -o bench/${i}.dat -t vector
done