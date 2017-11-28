# HDF5
HDF5_INCLUDE=/usr/include/hdf5/serial
HDF5_LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial/

# C compiler
CC=nvcc -gencode arch=compute_35,code=sm_35
CCFLAGS=-I. -I$(HDF5_INCLUDE) -L$(HDF5_LIB)

rt: rt.cu grid.cu
	$(CC) -o a.out rt.cu grid.cu $(CCFLAGS) -lm -lhdf5
