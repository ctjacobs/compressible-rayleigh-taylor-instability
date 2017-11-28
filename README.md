# Overview

This code solves the compressible Navier-Stokes equations using the finite difference method. It runs on an NVIDIA GPU using the CUDA framework.

# Compiling and running

Use the `make` command to compile the code. This step requires the `nvcc` compiler.

When the code is run, the solution fields are written in HDF5 format every 250 iterations.

# License

The code is released under the MIT license. See the file `LICENSE.md` for more information.
