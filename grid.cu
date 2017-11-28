// Copyright (C) 2017 Christian T. Jacobs

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
