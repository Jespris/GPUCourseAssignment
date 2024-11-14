#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;


int main(int argc, char *argv[])
{
   int    i;
   int    noofblocks;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long int histogramDRsum, histogramDDsum, histogramRRsum;
   double w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;

   FILE *outfil;

   // if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);
/*
   // if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

   // allocate memory on the GPU

   // copy data to the GPU

   // run the kernels on the GPU

   // copy the results back to the CPU

   // calculate omega values on the CPU, can of course be done on the GPU

   kerneltime = 0.0;
   gettimeofday(&_ttime, &_tzone);
   start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

// run your kernel here

   gettimeofday(&_ttime, &_tzone);
   end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
   kerneltime += end-start;
   printf("   Run time = %.lf secs\n",kerneltime);
*/
   return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec, phi, theta, dpi;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != deviceNo ) printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

