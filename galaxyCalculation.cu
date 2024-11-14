#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BIN_RANGE 180
#define BINS_PER_DEGREE 4
#define THREADS_PER_BLOCK 512

__global__ void calculateAngle(float ra_A, float decl_A, float* ra_B, float* decl_B, float* angles, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) {
		// I is larger than N, something has gone wrong
		// TODO: print an error here somehow
		return;
	}
	// convert angles from arc minutes to radians
	// TODO: save this constant somewhere else ?!?
	float arcmin_to_rad = 0.00029088820866;
	float rad_to_deg = 57.29577951;

	float alpha_A = ra_A*arcmin_to_rad;
	float delta_A = decl_A*arcmin_to_rad; 
	float alpha_B = ra_B[i]*arcmin_to_rad;
	float delta_B = decl_B[i]*arcmin_to_rad;

	// TODO: check that we have logical values

	float dotProduct = cos(delta_A)*cos(delta_B)*cos(alpha_A-alpha_B)+sin(delta_A)*sin(delta_B);
	float theta_rad = acos(dotProduct);
	
	float theta_deg = theta_rad * rad_to_deg;

	// TODO: check that we have logical result
	if (theta_deg < 0){
		theta_deg = abs(theta_deg);
	}
	// TODO: assert theta_deg <= 180
	angles[i] = theta_deg;
}

__global__ void fillHistogram(float* angles, unsigned int* histogram){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float angle = angles[i];
	int histogramBinIndex = floor(angle * BINS_PER_DEGREE);
	unsigned int *histogramAddess = &histogram[histogramBinIndex];
	atomicInc(histogramAddess, 1000000);
}

void summarize_histogram(unsigned int* histogram){
	long total_count = 0;
	int max_height = 0;
	int mode_bin = 0;
	int non_zero_start = -1, non_zero_end = -1;
	int NUM_BINS = BIN_RANGE * BINS_PER_DEGREE;

	// Calculate total count, find max height and mode, and locate non-zero range
    for (int i = 0; i < NUM_BINS; i++) {
        total_count += histogram[i];
        
        if (histogram[i] > max_height) {
            max_height = histogram[i];
            mode_bin = i;
        }
        
        if (histogram[i] > 0) {
            if (non_zero_start == -1) non_zero_start = i;
            non_zero_end = i;
        }
    }
    
    // Calculate mean bin height
    double mean_height = (double)total_count / NUM_BINS;
    
    // Calculate variance and standard deviation
    double variance = 0.0;
    for (int i = 0; i < NUM_BINS; i++) {
        variance += pow(histogram[i] - mean_height, 2);
    }
    variance /= NUM_BINS;
    double stddev = sqrt(variance);
    
    // Approximate median (finding the bin where cumulative count reaches 50%)
    long cumulative_count = 0;
    int median_bin = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        cumulative_count += histogram[i];
        if (cumulative_count >= total_count / 2) {
            median_bin = i;
            break;
        }
    }
    
    // Output summary
    printf("Histogram Summary:\n");
    printf("Total count: %ld\n", total_count);
    printf("Mean bin height: %.2f\n", mean_height);
    printf("Median bin (approx): %d\n", median_bin);
    printf("Mode bin: %d with height %d\n", mode_bin, max_height);
    printf("Standard deviation of bin heights: %.2f\n", stddev);
    printf("Non-zero bin range: %d to %d\n", non_zero_start, non_zero_end);
}

// data for the real galaxies will be read into these arrays
float *h_raReal, *h_declReal;
// number of real galaxies
int nrReal;

// data for the simulated random galaxies will be read into these arrays
float *h_raFake, *h_declFake;
// number of simulated random galaxies
int nrFake;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

int main(int argc, char *argv[])
{
	int i;
	int nrBlocks;
	printf("==================== READING INPUT DATA =====================\n");
	int readdata(char *argv1, char *argv2);
	printf("==================== READING DEVICE DATA ====================\n");
	int getDevice(int deviceNr);
	long int histogramDRsum, histogramDDsum, histogramRRsum;
	double w;
	double start, end, kerneltime;
	// struct timeval _ttime;
	// struct timezone _tzone;
	cudaError_t myError;

	FILE *outfil;

	printf("================= CALCULATING DR ANGLES =====================\n");

	if (argc != 4)
	{
		printf("Usage: a.out real_data random_data output_data\n");
		return (-1);
	}

	if (getDevice(0) != 0)
		return (-1);

	// TODO: SOMEHOW WE AREN*T GOING PAST THIS?!?!?!?
	if (readdata(argv[1], argv[2]) != 0)
		return (-1);

	// TODO: make sure input array sizes are the same
	// allocate memory on the GPU
	int N = nrReal;
	if (N != nrFake){
		printf("Input data lengths are not the same!");
		return (-1);
	}

	size_t arraybytes = N * sizeof(float);
	size_t histogrambytes = BIN_RANGE * BINS_PER_DEGREE * sizeof(unsigned int);
	unsigned int* h_histogram = (unsigned int*)malloc(histogrambytes);
	unsigned int* d_histogram; cudaMalloc(&d_histogram, histogrambytes);

	// DR histogram implementation
	for (int i=0; i<N; i++){
		clock_t start, end;
		double time_used;
		printf("Calculating angles for galaxy %d...\n", i);
		start = clock();
		// TODO: do we need to malloc the h_raReal, h_declReal... ?
		float* h_results = (float*)malloc(arraybytes);

		// allocate to GPU: the real and fake right ascension and declination, aswell as a result array

		float d_raReal = h_raReal[i];
		float d_declReal = h_declReal[i];
		
		float* d_raFake; cudaMalloc(&d_raFake, arraybytes);
		float* d_declFake; cudaMalloc(&d_declFake, arraybytes);
		float* d_results; cudaMalloc(&d_results, arraybytes);

		// copy data to the GPU
		cudaMemcpy(d_raFake, h_raFake, arraybytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_declFake, h_declFake, arraybytes, cudaMemcpyHostToDevice);

		// Size of thread blocks
		int blocksInGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		// run the kernels on the GPU
		// THIS CALCULATES DR
		printf("Calculating angles between real and fake...\n");
		calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_raReal, d_declReal, d_raFake, d_declFake, d_results, N);
		
		// copy the results back to the CPU
		cudaMemcpy(h_results, d_results, arraybytes, cudaMemcpyDeviceToHost);
		printf("DONE!\n");
		
		// Free memory
		cudaFree(d_declFake); cudaFree(d_raFake); cudaFree(d_results);
		
		// print some result values as testing
		printf("Result value of index 0 = %.2f\n", h_results[0]);
		printf("Result value of index 1 = %.2f\n", h_results[1]);
		printf("Result value of index 2 = %.2f\n", h_results[2]);
		printf("Result value of index 3 = %.2f\n", h_results[3]);
		printf("Result value of index %d = %.2f\n", N - 1, h_results[N - 1]);

		// Fill histogram
		float* d_angles; cudaMalloc(&d_angles, arraybytes);
		cudaMemcpy(d_angles, h_results, arraybytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_histogram, h_histogram, histogrambytes, cudaMemcpyHostToDevice);
		// Use same blocksInGrid, THREADS_PER_BLOCK
		printf("Incementing bins in histogram...\n");
		fillHistogram<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_angles, d_histogram);
		// Copy results
		cudaMemcpy(h_histogram, d_histogram, histogrambytes, cudaMemcpyDeviceToHost);
		printf("DONE!");
		// Free memory
		cudaFree(d_angles); cudaFree(d_histogram);
		free(h_results);

		end = clock();
		time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
		printf("Execution time: %.2f ms\n", time_used);
	}

	printf("=================== SUMMARIZING HISTOGRAM ===================\n");

	summarize_histogram(h_histogram);
	
	// calculate omega values on the CPU, can of course be done on the GPU

	return (0);
}

int readdata(char *argv1, char *argv2)
{
	int i, linecount;
	char inbuf[180];
	double ra, dec, phi, theta, dpi;
	FILE *infil;

	printf("   Assuming input data is given in arc minutes!\n");
	// spherical coordinates phi and theta in radians:
	// phi   = ra/60.0 * dpi/180.0;
	// theta = (90.0-dec/60.0)*dpi/180.0;

	dpi = acos(-1.0);
	infil = fopen(argv1, "r");
	if (infil == NULL)
	{
		printf("Cannot open input file %s\n", argv1);
		return (-1);
	}

	// read the number of galaxies in the input file
	int announcednumber;
	if (fscanf(infil, "%d\n", &announcednumber) != 1)
	{
		printf(" cannot read file %s\n", argv1);
		return (-1);
	}
	linecount = 0;
	while (fgets(inbuf, 180, infil) != NULL)
		++linecount;
	rewind(infil);

	if (linecount == announcednumber)
		printf("   %s contains %d galaxies\n", argv1, linecount);
	else
	{
		printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
		return (-1);
	}

	nrReal = linecount;
	h_raReal = (float *)calloc(nrReal, sizeof(float));
	h_declReal = (float *)calloc(nrReal, sizeof(float));

	// skip the number of galaxies in the input file
	if (fgets(inbuf, 180, infil) == NULL)
		return (-1);
	i = 0;
	while (fgets(inbuf, 80, infil) != NULL)
	{
		if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
		{
			printf("   Cannot read line %d in %s\n", i + 1, argv1);
			fclose(infil);
			return (-1);
		}
		h_raReal[i] = (float)ra;
		h_declReal[i] = (float)dec;
		++i;
	}

	fclose(infil);

	if (i != nrReal)
	{
		printf("   Cannot read %s correctly\n", argv1);
		return (-1);
	}

	infil = fopen(argv2, "r");
	if (infil == NULL)
	{
		printf("Cannot open input file %s\n", argv2);
		return (-1);
	}

	if (fscanf(infil, "%d\n", &announcednumber) != 1)
	{
		printf(" cannot read file %s\n", argv2);
		return (-1);
	}
	linecount = 0;
	while (fgets(inbuf, 80, infil) != NULL)
		++linecount;
	rewind(infil);

	if (linecount == announcednumber)
		printf("   %s contains %d galaxies\n", argv2, linecount);
	else
	{
		printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
		return (-1);
	}

	nrFake = linecount;
	h_raFake = (float *)calloc(nrFake, sizeof(float));
	h_declFake = (float *)calloc(nrFake, sizeof(float));

	// skip the number of galaxies in the input file
	if (fgets(inbuf, 180, infil) == NULL)
		return (-1);
	i = 0;
	while (fgets(inbuf, 80, infil) != NULL)
	{
		if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2)
		{
			printf("   Cannot read line %d in %s\n", i + 1, argv2);
			fclose(infil);
			return (-1);
		}
		h_raFake[i] = (float)ra;
		h_declFake[i] = (float)dec;
		++i;
	}

	fclose(infil);

	if (i != nrFake)
	{
		printf("   Cannot read %s correctly\n", argv2);
		return (-1);
	}

	return (0);
}

int getDevice(int deviceNo)
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("   Found %d CUDA devices\n", deviceCount);
	if (deviceCount < 0 || deviceCount > 128)
		return (-1);
	int device;
	for (device = 0; device < deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("      Device %s                  device %d\n", deviceProp.name, device);
		printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
		printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
		printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
		printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
		printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
		printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
		printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
		printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
		printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
		printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
		printf("         maxGridSize                   =   %d x %d x %d\n",
			   deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
			   deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("         concurrentKernels             =   ");
		if (deviceProp.concurrentKernels == 1)
			printf("     yes\n");
		else
			printf("    no\n");
		printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
		if (deviceProp.deviceOverlap == 1)
			printf("            Concurrently copy memory/execute kernel\n");
	}

	cudaSetDevice(deviceNo);
	cudaGetDevice(&device);
	if (device != deviceNo)
		printf("   Unable to set device %d, using device %d instead", deviceNo, device);
	else
		printf("   Using CUDA device %d\n\n", device);

	return (0);
}
