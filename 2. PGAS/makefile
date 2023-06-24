SHELL := /bin/bash

n ?= 5
file ?= matrix5.txt
# upcxx-nodes ?= $(upcxx-nodes nodes)

all: compile run clean

compile:
	source /opt/nfs/config/source_upcxx_2023.3.sh && \
	/home/nfs/config/station204_name_list.sh 1 16 > nodes && \
	UPCXX_GASNET_CONDUIT=udp upcxx -O2 main.cpp -o QR_decomp

run:
	source /opt/nfs/config/source_upcxx_2023.3.sh && upcxx-run -shared-heap 256M -n $(n) $$(upcxx-nodes nodes) QR_decomp $(file)

clean:
	rm -f QR_decomp; rm nodes
