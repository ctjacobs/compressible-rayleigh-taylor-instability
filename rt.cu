// Solves the compressible Navier-Stokes equations using the finite difference method. It runs on an NVIDIA GPU using the CUDA framework.
// The current setup is designed to simulate a 2D Rayleigh-Taylor instability, with periodic boundary conditions along the y planes and no-normal flow conditions along the x planes. A Lax-Friedrichs scheme is used to introduce artificial viscosity. See http://users.monash.edu/~sergiys/M43071.html for more information on the setup and the numerical methods used.

// Copyright (C) 2017 Christian T. Jacobs

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include "grid.h"

__device__ double dfdx(grid *g, double *f, int i, int j);
__device__ double dfdy(grid *g, double *f, int i, int j);
__device__ double d2fdx2(grid *g, double *f, int i, int j);
__device__ double d2fdy2(grid *g, double *f, int i, int j);
    
// Constants.
const double ga = -10.0;  // Acceleration due to gravity.
const double ratio_of_specific_heats = 1.4;  // Ratio of specific heats.

__host__
void initial_conditions(grid g, double *u, double *v, double *r, double *e, double *p, double *ru, double *rv)
{
    for (int j = 0; j < g.Nyh; j++)
    {
        for (int i = 0; i < g.Nxh; i++)
        {
            int width = g.Nxh;
            int offset = j*width + i;
            
            if(j*g.dy - g.dy - g.Ly/2.0 >= 0)
            {
                r[offset] = 2.0;
            }
            else
            {
                r[offset] = 1.0;
            }

            u[offset] = 0.0;

            // Add perturbations to v.
            if((j*g.dy - g.dy - g.Ly/2.0) >= -0.05 && (j*g.dy - g.dy - g.Ly/2.0 <= 0.05))
            {
                v[offset] = 1e-3*2.0*( (double)rand() / (double)RAND_MAX - 0.5);
            }
            else
            {
                v[offset] = 0.0;
            }

            p[offset] = 40 + r[offset]*ga*(j*g.dy - g.dy - g.Ly/2.0);
            e[offset] = p[offset]/(ratio_of_specific_heats-1.0) + 0.5*r[offset]*(pow(u[offset],2) + pow(v[offset],2));

            ru[offset] = r[offset]*u[offset];
            rv[offset] = r[offset]*v[offset];
        }
    }
    

    return;
}

__global__
void combined_fields(grid g, double *u, double *v, double *r, double *e, double *p, double *ru, double *rv, double *ruu, double *ruv, double *rvv, double *peu, double *pev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = j*g.Nxh + i;
    if(offset >= g.N)
       return;

    // Compute combined fields.
    p[offset] = (ratio_of_specific_heats-1.0)*(e[offset] - 0.5*r[offset]*(pow(u[offset],2) + pow(v[offset],2)));  // Equation of state.
    ruu[offset] = ru[offset]*u[offset];
    ruv[offset] = ru[offset]*v[offset];
    rvv[offset] = rv[offset]*v[offset];
    peu[offset] = u[offset]*(p[offset] + e[offset]);
    pev[offset] = v[offset]*(p[offset] + e[offset]);

    return;
}

__global__
void extract_fields(grid g, double *u, double *v, double *r, double *ru, double *rv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = j*g.Nxh + i;
    if(offset >= g.N)
       return;

    // Extract primitive variables.
    u[offset] = ru[offset]/r[offset];
    v[offset] = rv[offset]/r[offset];

    return;
}

__global__
void rhs(grid g, double *u, double *v, double *r, double *e, double *p, double *ru, double *rv, double *ruu, double *ruv, double *rvv, double *peu, double *pev, double k1, double k2, double k3, double *rhs_mass, double *rhs_momentum_x, double *rhs_momentum_y, double *rhs_energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = j*g.Nxh + i;

    // Don't consider indices that are out of bounds.
    if(offset >= g.N || i == 0 || i >= g.Nxh-1 || j == 0 || j >= g.Nyh-1)
       return;
    
    // Continuity equation's RHS.
    double drudx = dfdx(&g, ru, i, j);
    double drvdy = dfdy(&g, rv, i, j);
    rhs_mass[offset] = -(drudx + drvdy - k1*d2fdx2(&g, r, i, j) - k1*d2fdy2(&g, r, i, j));

    // Momentum equation's RHS (x direction).
    double druudx = dfdx(&g, ruu, i, j);
    double druvdy = dfdy(&g, ruv, i, j);
    double dpdx = dfdx(&g, p, i, j);
    rhs_momentum_x[offset] = -(druudx + druvdy + dpdx - k2*d2fdx2(&g, ru, i, j) );

    // Momentum equation's RHS (y direction).
    double drvvdy = dfdy(&g, rvv, i, j);
    double druvdx = dfdx(&g, ruv, i, j);
    double dpdy = dfdy(&g, p, i, j);
    rhs_momentum_y[offset] = -(drvvdy + druvdx + dpdy - ga*r[offset] - k2*d2fdy2(&g, rv, i, j) );

    // Energy equation's RHS.
    double dpeudx = dfdx(&g, peu, i, j);
    double dpevdy = dfdy(&g, pev, i, j); 
    rhs_energy[offset] = -(dpeudx + dpevdy - ga*r[offset]*v[offset] - k3*d2fdx2(&g, e, i, j) - k3*d2fdy2(&g, e, i, j) );

    return;
}

__global__
void advance(grid g, double *r, double *ru, double *rv, double *e, double *r_old, double *ru_old, double *rv_old, double *e_old, double *rhs_mass, double *rhs_momentum_x, double *rhs_momentum_y, double *rhs_energy, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = j*g.Nxh + i;
    if(offset >= g.N || i == 0 || i >= g.Nxh-1 || j == 0 || j >= g.Nyh-1)
       return;

    r[offset] = dt*rhs_mass[offset] + r_old[offset];
    ru[offset] = dt*rhs_momentum_x[offset] + ru_old[offset];
    rv[offset] = dt*rhs_momentum_y[offset] + rv_old[offset];
    e[offset] = dt*rhs_energy[offset] + e_old[offset];

    return;
}

__host__
void boundary_conditions(grid g, double *r, double *ru, double *rv, double *e, double *p)
{
    int width = g.Nxh;

    // x planes: no-normal flow conditions.
    for (int i = 0; i < g.Nxh; i++)
    {
        r[i] = r[width*2 + i];
        r[width*(g.Nyh-1) + i] = r[width*(g.Nyh-3) + i];
        
        ru[i] = ru[width*2 + i];
        ru[width*(g.Nyh-1) + i] = ru[width*(g.Nyh-3) + i];

        rv[i] = -rv[width*2 + i];
        rv[width*(g.Nyh-1) + i] = -rv[width*(g.Nyh-3) + i];

        p[i] = 40 + r[i]*-10*(- g.dy - g.Ly/2.0);
        p[width*(g.Nyh-1) + i] = 40 + r[width*(g.Nyh-1) + i]*-10*((g.Nyh-1)*g.dy - g.dy - g.Ly/2.0);

        e[i] = p[i]/(ratio_of_specific_heats-1.0) + 0.5*r[i]*(pow(ru[i]/r[i],2) + pow(rv[i]/r[i],2));
        e[width*(g.Nyh-1) + i] = p[width*(g.Nyh-1) + i]/(ratio_of_specific_heats-1.0) + 0.5*r[width*(g.Nyh-1) + i]*(pow(ru[width*(g.Nyh-1) + i]/r[width*(g.Nyh-1) + i],2) + pow(rv[width*(g.Nyh-1) + i]/r[width*(g.Nyh-1) + i],2));
    }


    return;
}

__host__
double max_dt(grid g, double *r, double *u, double *v, double *p)
{
    // Find the maximum timestep size.
    double velocity_max = 0.0;
    double m = 0.0;
    for (int j = 1; j < g.Nyh-1; j++)
    {
        for (int i = 1; i < g.Nxh-1; i++)
        {
            int width = g.Nxh;
            int offset = j*width + i;
            // Speed of sound needs to be considered for compressible flows.
            double speed_of_sound = sqrt(ratio_of_specific_heats*p[offset]/r[offset]);
            if(speed_of_sound > fabs(u[offset]) && speed_of_sound > fabs(v[offset]))
            {
                m = speed_of_sound;
            }
            else
            {
                if(fabs(u[offset]) > fabs(v[offset]))
                    m = fabs(u[offset]);
                else
                    m = fabs(v[offset]);
            }
            if(m > velocity_max)
                velocity_max = m;
        }
    }
    double courant_number = 0.2;
    return courant_number*(g.dx)/velocity_max;
}

__host__
void dump(int n, grid g, double *r, double *u, double *v, double *e)
{
    // HDF5 I/O.
    char h5fname[100];
    sprintf(h5fname, "fields_%d.h5", n);
    hid_t h5fid, h5gid, h5sid, h5did_r, h5did_u, h5did_v, h5did_e;
    hsize_t dims[1];
    herr_t h5status;
    
    // Create HDF5 file, group, and datasets.
    h5fid = H5Fcreate(h5fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    h5gid = H5Gcreate2(h5fid, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = g.N;
    h5sid = H5Screate_simple(1, dims, NULL);
    h5did_r = H5Dcreate2(h5fid, "/fields/r", H5T_IEEE_F64BE, h5sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5did_u = H5Dcreate2(h5fid, "/fields/u", H5T_IEEE_F64BE, h5sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5did_v = H5Dcreate2(h5fid, "/fields/v", H5T_IEEE_F64BE, h5sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5did_e = H5Dcreate2(h5fid, "/fields/e", H5T_IEEE_F64BE, h5sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Write fields.
    h5status = H5Dwrite(h5did_r, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, r);
    h5status = H5Dwrite(h5did_u, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u);
    h5status = H5Dwrite(h5did_v, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
    h5status = H5Dwrite(h5did_e, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, e);

    // Close HDF5 files.
    h5status = H5Dclose(h5did_r);
    h5status = H5Dclose(h5did_u);
    h5status = H5Dclose(h5did_v);
    h5status = H5Dclose(h5did_e);
    h5status = H5Sclose(h5sid);
    h5status = H5Gclose(h5gid);
    h5status = H5Fclose(h5fid); 
    
    return;
}

__device__
double dfdx(grid *g, double *f, int i, int j)
{
    int width = g->Nxh;
    // Periodic boundaries on the y planes.
    if(i==1)
    {
        return (0.5*f[j*width + (i+1)] - 0.5*f[j*width + (g->Nxh-2)])/g->dx;
    }
    else if(i==g->Nxh-2)
    {
        return (0.5*f[j*width + 1] - 0.5*f[j*width + (i-1)])/g->dx;
    }
    else
    {
        return (0.5*f[j*width + (i+1)] - 0.5*f[j*width + (i-1)])/g->dx;
    }
}

__device__
double dfdy(grid *g, double *f, int i, int j)
{
    int width = g->Nxh;    
    return (0.5*f[(j+1)*width + i] - 0.5*f[(j-1)*width + i])/g->dy;
}

__device__
double d2fdx2(grid *g, double *f, int i, int j)
{
    int width = g->Nxh;
    // Periodic boundaries on the y planes.
    if(i==1)
    {
        return (f[j*width + (i+1)] - 2.0*f[j*width + i] + f[j*width + (g->Nxh-2)])/pow(g->dx,2);
    }
    else if(i==g->Nxh-2)
    {
        return (f[j*width + 1] - 2.0*f[j*width + i] + f[j*width + (i-1)])/pow(g->dx,2);
    }
    else
    {
        return (f[j*width + (i+1)] - 2.0*f[j*width + i] + f[j*width + (i-1)])/pow(g->dx,2);
    }
}

__device__
double d2fdy2(grid *g, double *f, int i, int j)
{
    int width = g->Nxh;
    return (f[(j+1)*width + i] - 2.0*f[j*width + i] + f[(j-1)*width + i])/pow(g->dy,2);
}

int main()
{
    // Timestepping.
    double t = 0.0;
    double dt;
    int Nt = 200000;

    // Solution arrays.
    double *r, *u, *v, *p, *e, *ru, *rv;
    double *r_old, *ru_old, *rv_old, *e_old;
    double *ruu, *rvv, *ruv, *peu, *pev;
    double *rhs_mass, *rhs_momentum_x, *rhs_momentum_y, *rhs_energy;

    // Grid.
    grid g;
    
    // Initialise grid.
    initialise_grid(&g);

    // Conservative variables.
    cudaMallocManaged(&r, g.N*sizeof(double));
    cudaMallocManaged(&ru, g.N*sizeof(double));
    cudaMallocManaged(&rv, g.N*sizeof(double));
    cudaMallocManaged(&u, g.N*sizeof(double));
    cudaMallocManaged(&v, g.N*sizeof(double));
    cudaMallocManaged(&e, g.N*sizeof(double));
    cudaMallocManaged(&p, g.N*sizeof(double));
    cudaMallocManaged(&r_old, g.N*sizeof(double));
    cudaMallocManaged(&ru_old, g.N*sizeof(double));
    cudaMallocManaged(&rv_old, g.N*sizeof(double));
    cudaMallocManaged(&e_old, g.N*sizeof(double));

    // Temporary variables.  
    cudaMallocManaged(&ruu, g.N*sizeof(double));
    cudaMallocManaged(&ruv, g.N*sizeof(double));
    cudaMallocManaged(&rvv, g.N*sizeof(double));
    cudaMallocManaged(&peu, g.N*sizeof(double));
    cudaMallocManaged(&pev, g.N*sizeof(double));
    
    // RHS arrays.
    cudaMallocManaged(&rhs_mass, g.N*sizeof(double));
    cudaMallocManaged(&rhs_momentum_x, g.N*sizeof(double));
    cudaMallocManaged(&rhs_momentum_y, g.N*sizeof(double));
    cudaMallocManaged(&rhs_energy, g.N*sizeof(double));

    // CUDA block and thread parameters.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((g.Nxh + g.Nxh-1)/ threadsPerBlock.x, (g.Nyh + g.Nyh-1) / threadsPerBlock.y);

    // Initial conditions.
    initial_conditions(g, u, v, r, e, p, ru, rv);

    // Apply boundary condition.           
    boundary_conditions(g, r, ru, rv, e, p);
    
    // Write initial state.
    dump(0, g, r, u, v, e);

    // Forward Euler timestepping scheme.
    for (int n = 1; n <= Nt; n++)
    {
        printf("Timestep %d (t = %g)\n", n, t);
        
        // Apply boundary condition.
        boundary_conditions(g, r, ru, rv, e, p);

        // Save the state of the fields at the previous timestep.
        memcpy(r_old, r, g.N*sizeof(double));
        memcpy(ru_old, ru, g.N*sizeof(double));
        memcpy(rv_old, rv, g.N*sizeof(double));
        memcpy(e_old, e, g.N*sizeof(double));

        // Extract the primitive variables and compute the combined fields.
        extract_fields<<<numBlocks, threadsPerBlock>>>(g, u, v, r, ru, rv);
        cudaDeviceSynchronize();
        combined_fields<<<numBlocks, threadsPerBlock>>>(g, u, v, r, e, p, ru, rv, ruu, ruv, rvv, peu, pev);
        cudaDeviceSynchronize();

        // Compute maximum value of dt.
        dt = max_dt(g, r, u, v, p);
        t += dt;

        // Diffusivity constants for the Lax-Friedrichs scheme.
        double D = pow(g.dx,2)/(2.0*dt);
        double k1 = 0.0125*D;
        double k2 = 0.125*D;
        double k3 = 0.0125*D;

        // Compute RHSs.
        rhs<<<numBlocks, threadsPerBlock>>>(g, u, v, r, e, p, ru, rv, ruu, ruv, rvv, peu, pev, k1, k2, k3, rhs_mass, rhs_momentum_x, rhs_momentum_y, rhs_energy);
        cudaDeviceSynchronize();

        // Advance the variables in time.
        advance<<<numBlocks, threadsPerBlock>>>(g, r, ru, rv, e, r_old, ru_old, rv_old, e_old, rhs_mass, rhs_momentum_x, rhs_momentum_y, rhs_energy, dt);
        cudaDeviceSynchronize();
        
        // Write primitive variables to HDF5 file every 250 iterations.
        if(n % 250 == 0)
        {
            extract_fields<<<numBlocks, threadsPerBlock>>>(g, u, v, r, ru, rv);
            cudaDeviceSynchronize();
            dump(n, g, r, u, v, e);
        }
    }
 
    // Free allocated memory.
    cudaFree(r);
    cudaFree(u);
    cudaFree(v);
    cudaFree(ru);
    cudaFree(rv);
    cudaFree(e);
    cudaFree(p);
    cudaFree(r_old);
    cudaFree(ru_old);
    cudaFree(rv_old);
    cudaFree(e_old);
    cudaFree(rhs_mass);
    cudaFree(rhs_momentum_x);
    cudaFree(rhs_momentum_y);
    cudaFree(rhs_energy);
    cudaFree(ruu);
    cudaFree(rvv);
    cudaFree(peu);
    cudaFree(pev);

    return 0;
}
