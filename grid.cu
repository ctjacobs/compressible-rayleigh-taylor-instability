#include <stdlib.h>
#include "grid.h"

void initialise_grid(grid *g)
{   
    g->Lx = 1.0;
    g->Ly = 3.0;

    g->Nx = 1022;
    g->Ny = 1022;
    g->Nh = 1;
    g->Nxh = g->Nx + 2*g->Nh;
    g->Nyh = g->Ny + 2*g->Nh;
    g->N = (g->Nxh)*(g->Nyh);

    g->dx = g->Lx/(g->Nxh-3);
    g->dy = g->Ly/(g->Nyh-3);

    return;
}
