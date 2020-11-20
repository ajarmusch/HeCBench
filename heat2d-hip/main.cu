#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "defs.h"
//#include "io.h"
//#include "lapl_ss.h"

/* Choose the optimized gpu kernel */
#define OPT

#include "io.c"
#include "lapl_ss.c"

double
stop_watch(double t0) 
{
  double time;
  struct timeval t;
  gettimeofday(&t, NULL);
  time = (double) t.tv_sec + (double) t.tv_usec * 1e-6;  
  return time-t0;
}

void
usage(char *argv[]) {
  fprintf(stderr, " Usage: %s LX LY NITER IN_FILE\nIN_FILE can be generated by python mkinit LX LY IN_FILE\n", 
		  argv[0]);
  return;
}

/*
 * Naive implementation of a single iteration of the lapl
 * equation. Each thread takes one site of the output array
 */
__global__ void
dev_lapl_iter(float *out, const float *in, const float delta, const float norm, const int lx, const int ly)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int x = i % lx;
  int y = i / lx;
  int v00 = y*lx + x;
  int v0p = y*lx + (x + 1)%lx;
  int v0m = y*lx + (lx + x - 1)%lx;
  int vp0 = ((y+1)%ly)*lx + x;
  int vm0 = ((ly+y-1)%ly)*lx + x;
  out[v00] = norm*in[v00]
    + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
  return;
}


int Lx, Ly;

int
main(int argc, char *argv[]) {
  /* Check the number of command line arguments */
  if(argc != 5) {
    usage(argv);
    exit(1);
  }
  /* The length of the array in x and y is read from the command
     line */
  Lx = atoi(argv[1]);
  Ly = atoi(argv[2]);
  if (Lx % NTX != 0 ||  Ly % NTY != 0) {
    printf("Array length LX and LY must be a multiple of block size %d and %d, respectively\n", 
          NTX, NTY);
    exit(1);
  }
  /* The number of iterations */
  int niter = atoi(argv[3]);
  assert(niter >= 1);

  /* Fixed "sigma" */
  float sigma = 0.01;
  printf(" Ly,Lx = %d,%d\n", Ly, Lx);
  printf(" niter = %d\n", niter);
  printf(" input file = %s\n", argv[4]);
  /* Allocate the buffer for the data */
  float *cpu_arr = (float*) malloc(sizeof(float)*Lx*Ly);
  /* read file to buffer */
  read_from_file(cpu_arr, argv[4]);
  /* allocate super-site buffers */
  supersite *ssarr[2];
  posix_memalign((void**)&ssarr[0], 16, sizeof(supersite)*Lx*Ly/4);
  posix_memalign((void**)&ssarr[1], 16, sizeof(supersite)*Lx*Ly/4);
  /* convert input array to super-site packed */
  to_supersite(ssarr[0], cpu_arr);
  /* do iterations, record time */
  double t0 = stop_watch(0);
  for(int i=0; i<niter; i++) {
    lapl_iter_supersite(ssarr[(i+1)%2], sigma, ssarr[i%2]);
  }
  t0 = stop_watch(t0)/(double)niter;
  from_supersite(cpu_arr, ssarr[niter%2]);

  /* write to file */
  //write_to_file(fname, arr);
  /* write timing info */
  printf(" iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
	 niter, Lx, Ly, t0*1e6, 
	 Lx*Ly*sizeof(float)*2.0/(t0*1.0e9), 
	 (Lx*Ly*6.0)/(t0*1.0e9));
  /* free super-site buffers */
  for(int i=0; i<2; i++) {
    free(ssarr[i]);
  }
  /*
   * GPU part
   */

  /* read file again for GPU run */
  float *gpu_arr = (float*) malloc(sizeof(float)*Lx*Ly);
  read_from_file(gpu_arr, argv[4]);

  float *in, *out;	/* GPU arrays */

  /* Initialize: allocate GPU arrays and load array to GPU */
  hipMalloc((void **)&in, sizeof(float)*Lx*Ly);
  hipMalloc((void **)&out, sizeof(float)*Lx*Ly);
  hipMemcpy(in, gpu_arr, sizeof(float)*Lx*Ly, hipMemcpyHostToDevice);
  float xdelta = sigma / (1.0+4.0*sigma);
  float xnorm = 1.0/(1.0+4.0*sigma);

  /* Do iterations on GPU, record time */
  t0 = stop_watch(0);

  /* Fixed number of threads per block (in x- and y-direction), number
     of blocks per direction determined by dimensions Lx, Ly */
  dim3 blk(Lx/NTX * Ly/NTY);
  dim3 thr(NTX*NTY);
  for(int i=0; i<niter; i++) {
    hipLaunchKernelGGL(dev_lapl_iter, blk, thr, 0, 0, out, in, xdelta, xnorm, Lx, Ly);
    float* tmp = out;
    out = in;
    in = tmp;
  }
  hipDeviceSynchronize();
  t0 = stop_watch(t0)/(double)niter;

  /* copy GPU array to main memory and free GPU arrays */
  hipMemcpy(gpu_arr, in, sizeof(float)*Lx*Ly, hipMemcpyDeviceToHost);
  hipFree(in);
  hipFree(out);

  printf("Device: iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
  	 niter, Lx, Ly, t0*1e6,
  	 Lx*Ly*sizeof(float)*2.0/(t0*1.0e9),
  	 (Lx*Ly*6.0)/(t0*1.0e9));

  // verification
  for (int i = 0; i < Lx*Ly; i++) {
    // choose 1e-2 because the error rate increases with the iteration from 1 to 100000
    if ( fabs(cpu_arr[i] - gpu_arr[i]) > 1e-2 ) {
	    printf("FAILED at %d cpu=%f gpu=%f\n", i, cpu_arr[i], gpu_arr[i]);
            /* free main memory array */
            free(cpu_arr);
            free(gpu_arr);
	    return 1;
    }
  }

  /* write to file */
  //write_to_file(fname, arr);
  /* write timing info */
  free(cpu_arr);
  free(gpu_arr);
  return 0;
}
