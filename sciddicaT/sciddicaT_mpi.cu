// 1.43s (questo) vs 1.65s (monolithic)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "mpi.h"
#include "util.hpp"
#include <stdio.h>
// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

#define TILE_WIDTH 8
#define HALO_LENGTH 2

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char *path, int &nrows, int &ncols, double &nodata)
{
	FILE *f;

	if ((f = fopen(path, "r")) == 0)
	{
		printf("%s configuration header file not found\n", path);
		exit(0);
	}

	//Reading the header
	char str[STRLEN];
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str);
	ncols = atoi(str); //ncols
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str);
	nrows = atoi(str); //nrows
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str); //xllcorner = atof(str);  //xllcorner
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str); //yllcorner = atof(str);  //yllcorner
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str); //cellsize = atof(str);   //cellsize
	fscanf(f, "%s", &str);
	fscanf(f, "%s", &str);
	nodata = atof(str); //NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
	FILE *f = fopen(path, "r");

	if (!f)
	{
		printf("%s grid file not found\n", path);
		exit(0);
	}

	char str[STRLEN];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < columns; j++)
		{
			fscanf(f, "%s", str);
			SET(M, columns, i, j, atof(str));
		}

	fclose(f);

	return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
	FILE *f;
	f = fopen(path, "w");

	if (!f)
		return false;

	char str[STRLEN];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			sprintf(str, "%f ", GET(M, columns, i, j));
			fprintf(f, "%s ", str);
		}
		fprintf(f, "\n");
	}

	fclose(f);

	return true;
}

double *addLayer2D(int rows, int columns)
{
	double *tmp = (double *)malloc(sizeof(double) * rows * columns);
	if (!tmp)
		return NULL;
	return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
void simulationInit(int i, int j, int r, int c, double *Sz_h, double *Sh_h)
{
	double z, h;
	h = GET(Sh_h, c, i, j);

	if (h > 0.0)
	{
		z = GET(Sz_h, c, i, j);
		SET(Sz_h, c, i, j, z - h);
	}
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

__global__ void resetFlowsParallelized(int proc_id, int r, int c, int i_start, int i_end, double *Sf_h)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (proc_id == 1)
		i += 305;

	if (i >= i_start && i < i_end && j >= 1 && j < c)
	{
		BUF_SET(Sf_h, r, c, 0, i, j, 0.0);
		BUF_SET(Sf_h, r, c, 1, i, j, 0.0);
		BUF_SET(Sf_h, r, c, 2, i, j, 0.0);
		BUF_SET(Sf_h, r, c, 3, i, j, 0.0);
	}
}

__global__ void flowsComputationParallelized(int proc_id, int r, int c, int i_start, int i_end, double *Sz_h, double *Sh_h, double *Sf_h, double *halo_Sz, double *halo_Sh, double p_r, double p_epsilon)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= i_start && i < i_end && j >= 1 && j < c)
	{
		// Up, Left, Right, Down

		int tx = threadIdx.x;
		int ty = threadIdx.y;

		__shared__ double Sz_shared[TILE_WIDTH][TILE_WIDTH];
		__shared__ double Sh_shared[TILE_WIDTH][TILE_WIDTH];

		// Carico l'elemento principale nel tile
		Sz_shared[ty][tx] = GET(Sz_h, c, i, j);
		Sh_shared[ty][tx] = GET(Sh_h, c, i, j);
		__syncthreads();

		bool eliminated_cells[5] = {false, false, false, false, false};
		bool again;
		int cells_count;
		double average;
		double m;
		double u[5];
		int n;
		double z, h;

		m = Sh_shared[ty][tx] - p_epsilon;
		u[0] = Sz_shared[ty][tx] + p_epsilon;

		// UP
		if (i == 0)
		{
			z = halo_Sz[j];
			h = halo_Sh[j];
		}
		else
		{
			if (ty > 0)
			{ // Fetch from shared
				z = Sz_shared[ty - 1][tx];
				h = Sh_shared[ty - 1][tx];
			}
			else
			{ // Fetch from global
				z = GET(Sz_h, c, i - 1, j);
				h = GET(Sh_h, c, i - 1, j);
			}
		}
		u[1] = z + h;

		// LEFT

		if (tx > 0)
		{ // Fetch from shared
			z = Sz_shared[ty][tx - 1];
			h = Sh_shared[ty][tx - 1];
		}
		else
		{ // Fetch from global
			z = GET(Sz_h, c, i, j - 1);
			h = GET(Sh_h, c, i, j - 1);
		}
		u[2] = z + h;

		// RIGHT

		if (tx <= blockDim.x - 2)
		{ // Fetch from shared
			z = Sz_shared[ty][tx + 1];
			h = Sh_shared[ty][tx + 1];
		}
		else
		{ // Fetch from global
			z = GET(Sz_h, c, i, j + 1);
			h = GET(Sh_h, c, i, j + 1);
		}
		u[3] = z + h;

		// DOWN
		if (i == r / 2 - 1)
		{
			z = halo_Sz[j];
			h = halo_Sh[j];
		}
		else
		{
			if (ty <= blockDim.y - 2)
			{ // Fetch from shared
				z = Sz_shared[ty + 1][tx];
				h = Sh_shared[ty + 1][tx];
			}
			else
			{ // Fetch from global
				z = GET(Sz_h, c, i + 1, j);
				h = GET(Sh_h, c, i + 1, j);
			}
		}
		u[4] = z + h;

		do
		{
			again = false;
			average = m;
			cells_count = 0;

			for (n = 0; n < 5; n++)
				if (!eliminated_cells[n])
				{
					average += u[n];
					cells_count++;
				}

			if (cells_count != 0)
				average /= cells_count;

			for (n = 0; n < 5; n++)
				if ((average <= u[n]) && (!eliminated_cells[n]))
				{
					eliminated_cells[n] = true;
					again = true;
				}
		} while (again);


		if (proc_id == 1)
			i += 305;
		if (!eliminated_cells[1])
		{
			if (proc_id == 1)
				printf("P1 set");
			BUF_SET(Sf_h, r, c, 0, i, j, (average - u[1]) * p_r);
			int iLinear = (int)ceil((r * 2 * c * 0 + i * c) / 496.0f);
			// *modified = true;

			// printf("(%d, %d) <- T(%d, %d)\n", iLinear, j, i, j);
		}
		if (!eliminated_cells[2])
		{
			if (proc_id == 1)
				printf("P1 set");
			BUF_SET(Sf_h, r, c, 1, i, j, (average - u[2]) * p_r);
			int iLinear = (int)ceil((r * 2 * c * 1 + i * c) / 496.0f);
			// printf("(%d, %d) <- T(%d, %d)\n", iLinear, j, i, j);
			// *modified = true;
		}
		if (!eliminated_cells[3])
		{
			if (proc_id == 1)
				printf("P1 set");
			BUF_SET(Sf_h, r, c, 2, i, j, (average - u[3]) * p_r);
			int iLinear = (int)ceil((r * c * 2 + i * c) / 496.0f);
			// printf("(%d, %d) <- T(%d, %d)\n", iLinear, j, i, j);
			// *modified = true;
		}
		if (!eliminated_cells[4])
		{
			if (proc_id == 1)
				printf("P1 set");
			BUF_SET(Sf_h, r, c, 3, i, j, (average - u[4]) * p_r);
			int iLinear = (int)ceil((r * c * 3 + i * c) / 496.0f);
			// printf("(%d, %d) <- T(%d, %d)\n", iLinear, j, i, j);
			// *modified = true;
		}
	}
}

__global__ void widthUpdateParallelized(int proc_id, int r, int c, int i_start, int i_end, double *Sh_h, double *Sf_h)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i_offset = i;
	if (proc_id == 1)
		i += 305;

	__shared__ double Sf_shared[TILE_WIDTH][TILE_WIDTH][ADJACENT_CELLS];
	Sf_shared[ty][tx][0] = BUF_GET(Sf_h, r, c, 0, i_offset, j);
	Sf_shared[ty][tx][1] = BUF_GET(Sf_h, r, c, 1, i_offset, j);
	Sf_shared[ty][tx][2] = BUF_GET(Sf_h, r, c, 2, i_offset, j);
	Sf_shared[ty][tx][3] = BUF_GET(Sf_h, r, c, 3, i_offset, j);
	__syncthreads();

	if (i >= i_start && i < i_end && j >= 1 && j < c)
	{
		double h_next;
		h_next = GET(Sh_h, c, i, j);

		// accessi safe
		double b0 = Sf_shared[ty][tx][0];
		double b1 = Sf_shared[ty][tx][1];
		double b2 = Sf_shared[ty][tx][2];
		double b3 = Sf_shared[ty][tx][3];

		// accessi unsafe

		double bUp;
		if (ty > 0)
		{ // Fetch from shared
			bUp = Sf_shared[ty - 1][tx][3];
		}
		else if (ty == 0)
		{ // Fetch from global
			bUp = BUF_GET(Sf_h, r, c, 3, i_offset - 1, j);
		}

		double bLeft;
		if (tx > 0)
		{ // Fetch from shared
			bLeft = Sf_shared[ty][tx - 1][2];
		}
		else if (tx == 0)
		{ // Fetch from global
			bLeft = BUF_GET(Sf_h, r, c, 2, i_offset, j - 1);
		}

		double bRight;
		if (tx <= blockDim.x - 2)
		{ // Fetch from shared
			bRight = Sf_shared[ty][tx + 1][1];
		}
		else if (tx == blockDim.x - 1)
		{ // Fetch from global
			bRight = BUF_GET(Sf_h, r, c, 1, i_offset, j + 1);
		}

		double bDown;
		if (ty <= blockDim.y - 2)
		{ // Fetch from shared
			bDown = Sf_shared[ty + 1][tx][0];
		}
		else if (ty == blockDim.y - 1)
		{ // Fetch from global
			bDown = BUF_GET(Sf_h, r, c, 0, i_offset + 1, j);
		}

		// halo cells
		h_next += bUp - b0;	// UP
		h_next += bLeft - b1;  // LEFT
		h_next += bRight - b2; // RIGHT
		h_next += bDown - b3;  // DOWN

		SET(Sh_h, c, i, j, h_next);
	}
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	int rows, cols;
	double nodata;
	readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

	int r = rows;					  // r: grid rows
	int c = cols;					  // c: grid columns
	double p_r = P_R;				  // p_r: minimization algorithm outflows dumping factor
	double p_epsilon = P_EPSILON;	 // p_epsilon: frictional parameter threshold
	int steps = atoi(argv[STEPS_ID]); // steps: simulation steps
	int proc_id, num_procs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	cudaSetDevice(proc_id);
	int r_per_proc = r / num_procs;

	/* Pointers */
	double *Sz_h;
	double *Sh_h;
	double *halo_Sz;
	double *halo_Sh;
	double *Sz_d;
	double *Sh_d;
	double *Sf_d;

	/* Master allocation of Sz and Sh */
	if (proc_id == 0)
	{
		Sz_h = addLayer2D(r, c);					  // Allocates the Sz_h substate grid
		Sh_h = addLayer2D(r, c);					  // Allocates the Sh_h substate grid
		loadGrid2D(Sz_h, r, c, argv[DEM_PATH_ID]);	// Load Sz_h from file
		loadGrid2D(Sh_h, r, c, argv[SOURCE_PATH_ID]); // Load Sh_h from file
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				simulationInit(i, j, r, c, Sz_h, Sh_h);
	}

	/* Allocation of Cuda Elements */
	cudaMallocManaged(&Sz_d, sizeof(double) * r_per_proc * c);
	cudaMallocManaged(&Sh_d, sizeof(double) * r_per_proc * c);
	cudaMallocManaged(&Sf_d, sizeof(double) * r * c * ADJACENT_CELLS);
	cudaMallocManaged(&halo_Sz, c * sizeof(double));
	cudaMallocManaged(&halo_Sh, c * sizeof(double));

	/* Scattering of worker portions */
	MPI_Scatter(Sz_h, r_per_proc * c, MPI_DOUBLE, Sz_d, r_per_proc * c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(Sh_h, r_per_proc * c, MPI_DOUBLE, Sh_d, r_per_proc * c, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* Setting boundaries */
	int i_start, i_end;
	if (proc_id == 0)
	{
		i_start = 1;
		i_end = r_per_proc;
	}
	else if (proc_id == 1)
	{
		i_start = 0;
		i_end = r_per_proc - 1;
	}

	util::Timer cl_timer;
	// simulation loop

	double dimB = 8.0f;
	dim3 dimGrid(ceil(c / dimB), ceil(r_per_proc / dimB), 1);
	dim3 dimBlock(dimB, dimB, 1);

	printf("colonne: %d\n", c);
	printf("Grid: %dx%d\n", dimGrid.x, dimGrid.y);
	printf("Block: %dx%d\n", dimBlock.x, dimBlock.y);

	for (int s = 0; s < steps; ++s)
	{
		resetFlowsParallelized<<<dimGrid, dimBlock>>>(proc_id, i_start, i_end, r, c, Sf_d);
		cudaDeviceSynchronize();

		if (proc_id == 0)
		{
			MPI_Recv(halo_Sh, c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(Sh_d, c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(halo_Sz, c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(Sz_d, c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Send(Sh_d + ((r_per_proc - 1) * c), c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(halo_Sh, c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(Sz_d + ((r_per_proc - 1) * c), c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Recv(halo_Sz, c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		flowsComputationParallelized<<<dimGrid, dimBlock>>>(proc_id, r, c, i_start, i_end, Sz_d, Sh_d, Sf_d, halo_Sh, halo_Sz, p_r, p_epsilon);
		cudaDeviceSynchronize();

		widthUpdateParallelized<<<dimGrid, dimBlock>>>(proc_id, r, c, i_start, i_end, Sh_d, Sf_d);
		cudaDeviceSynchronize();
	}
	double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;

	printf("Elapsed time: %lf [s]\n", cl_time);

	double *tmp;
	if (proc_id == 0)
		tmp = (double *)malloc(sizeof(double) * r * c);

	MPI_Gather(Sh_d, r_per_proc * c, MPI_DOUBLE, tmp, r_per_proc * c, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (proc_id == 0)
		saveGrid2Dr(tmp, r, c, argv[OUTPUT_PATH_ID]); // Save Sh_h to file

	printf("Releasing memory...\n");
	cudaFree(Sh_d);
	cudaFree(Sz_d);
	cudaFree(Sf_d);
	cudaFree(halo_Sz);
	cudaFree(halo_Sh);
	if (proc_id == 0)
	{
		delete[] Sz_h;
		delete[] Sh_h;
	}

	MPI_Finalize();

	return 0;
}
