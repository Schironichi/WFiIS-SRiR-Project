SHELL := /bin/bash

n ?= 5
file ?= matrix5.txt

all: compile run clean

compile:
	source /opt/nfs/config/source_mpich411.sh && \
	/home/nfs/config/station204_name_list.sh 1 16 > nodes && \
	mpicxx main.cpp -o QR_decomp

run:
	source /opt/nfs/config/source_mpich411.sh && mpiexec -f nodes -n $(n) ./QR_decomp $(file)

clean:
	rm -f QR_decomp; rm nodes
