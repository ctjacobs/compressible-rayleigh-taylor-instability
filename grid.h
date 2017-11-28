typedef struct grid {
    int Nh; // Number of halo points at each boundary.
    int Nx, Ny; // Number of internal points.
    int Nxh, Nyh; // Number of internal points + 2*Nh
    int N; // Total number of grid points.
    double Lx, Ly;
    double dx, dy;
    
} grid;

void initialise_grid(grid *);
