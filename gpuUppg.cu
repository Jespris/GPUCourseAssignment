#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>  // For std::sort
#include <vector>     // For std::vector and std::pair
#include <limits.h>
#include <fstream>  // Include for file output
#include <iomanip> // Required for std::fixed and std::setprecision

#define DEGREE_RANGE 180 
#define BINS_PER_DEGREE 4
#define THREADS_PER_BLOCK 256
#define RAD_TO_DEG 57.29577951f

__global__ void calculateAngle(float* ra_A, float* decl_A, float* ra_B, float* decl_B, int* histogram, int N){
	float theta_deg;
	
	int tid = (blockDim.x * blockIdx.x + threadIdx.x);  // thread ID
	int k = tid % N;  // index of B
	int i = (tid - k) / N;  // index of A 
	
	/*
	if (i > N || k > N) {  // check that we're in bounds
		printf("Not in bounds\n");
		return;
	}
	*/

	// helper variables
	float alpha_A = ra_A[i];
	float delta_A = decl_A[i];
	float alpha_B = ra_B[k];
	float delta_B = decl_B[k];

	// TODO: check that we have logical values
	// printf("A1: %f, A2: %f, D1: %f, D2: %f\n", alpha_A, alpha_B, delta_A, delta_B);

	float dotProduct = sin(delta_A)*sin(delta_B) + cos(delta_A)*cos(delta_B)*cos(alpha_A - alpha_B);
	// printf("DotProduct: %.2f\n", dotProduct);
	// dotProduct = fminf(1.0f, fmaxf(dotProduct, -1.0f)); // Clamp to [-1, 1] due to floating point erros

	float theta_rad = acosf(dotProduct);  // Single precision will suffice
	theta_deg = theta_rad * RAD_TO_DEG;
	// printf("Theta in degrees: %.2f\n", theta_deg);
	// Remove floating point errors by clamping to [0, 180]
	theta_deg = fminf(179.999f, fmaxf(theta_deg, 0.0f)); 

	// histogram time!
	int histogramBinIndex = theta_deg * BINS_PER_DEGREE; // Get the index by multiplying the angle by bins per degree
	// printf("Bin index for %.2f degrees: %d\n", theta_deg, histogramBinIndex);

	if (histogramBinIndex >= 0 && histogramBinIndex < (DEGREE_RANGE * BINS_PER_DEGREE)){ // ensure boundary
		// TODO: maybe change to atomicInc, but then we need to change to size_t type arrays or something
		atomicAdd(&histogram[histogramBinIndex], 1);  // incementing histograms now, probably slows down the execution time a lot, 
		// but we don't have to launch another GPU program
	} else {
		printf("Outside bin range!\n");
	}
}

void verbose_omega(float* omega){
	printf("\nThe first thirty omega bins: \n");
	for (int i=0; i < 30; i++){
		printf("Bin [%d] value: 	 %.5f\n", i, omega[i]);
	}
}

void verbose_histogram(int* histogram) {
    long total_count = 0;
    int max_height = 0;
    int mode_bin = 0;
    int non_zero_start = -1, non_zero_end = -1;
    int num_bins = DEGREE_RANGE * BINS_PER_DEGREE;

    // Calculate total count, find max height and mode, and locate non-zero range
    for (int i = 0; i < num_bins; i++) {
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
    
    // Mean and standard deviation
    double mean_height = (double)total_count / num_bins;
    double variance = 0.0;
    for (int i = 0; i < num_bins; i++) {
        variance += pow(histogram[i] - mean_height, 2);
    }
    variance /= num_bins;
    double stddev = sqrt(variance);

	printf("\nThe first five bins: \n");
	for (int i=0; i < 5; i++){
		printf("Bin [%d] count: 	 %d\n", i, histogram[i]);
	}

    // Output summary
    printf("\nHistogram Summary:\n");
    printf("Total count: %ld\n", total_count);
    printf("Mean bin height: %.2f\n", mean_height);
    printf("Mode bin: %d with height %d\n", mode_bin, max_height);
    printf("Standard deviation of bin heights: %.2f\n", stddev);
    printf("Non-zero bin range: %d to %d\n", non_zero_start, non_zero_end);
}

void save_histogram_to_file(const int* histogram, int num_bins, const char* filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        printf("Error opening file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < num_bins; i++) {
        outfile << i << " " << histogram[i] << "\n";
    }
    outfile.close();
    printf("Histogram saved to %s\n", filename);
}


// data for the real galaxies will be read into these arrays
float *h_phiReal, *h_thetaReal;
// number of real galaxies
int nrReal;

// data for the simulated random galaxies will be read into these arrays
float *h_phiFake, *h_thetaFake;
// number of simulated random galaxies
int nrFake;

int main(int argc, char *argv[]) {
	printf("==================== READING INPUT DATA =====================\n");
	int readdata(char *argv1, char *argv2);
	printf("==================== READING DEVICE DATA ====================\n");
	int getDevice(int deviceNr);

	if (argc != 4){
		printf("Usage: a.out real_data random_data output_data\n");
		return (-1);
	}

	if (getDevice(0) != 0){
		printf("Failed finding a device!");
		return (-1);
	}
		

	if (readdata(argv[1], argv[2]) != 0){
		printf("Failed reading data!");
		return (-1);
	}
		
	// make sure input array sizes are the same
	
	int N = nrReal;
	if (N != nrFake){
		printf("Input data lengths are not the same! Exit...");
		return (-1);
	}

	printf("N = %d\n", N);

	clock_t start, end;
	double time_used;
	start = clock();

	long int totalPairs = (long int)N*N;

	// allocate memory on the GPU and histogram arrays memory on CPU
	size_t arraybytes = N * sizeof(float);
	size_t megabytes = (size_t)round((double)arraybytes / (1024 * 1024));
	printf("Arraybytes: %zd = %zd MB\n", arraybytes, megabytes);

	size_t histogrambytes = DEGREE_RANGE * BINS_PER_DEGREE * sizeof(int);
	int* h_histogramDR = (int*)malloc(histogrambytes);
	int* h_histogramDD = (int*)malloc(histogrambytes);
	int* h_histogramRR = (int*)malloc(histogrambytes);
	
	// allocate to GPU: the real and fake right ascension and declination
	float* d_phiReal; cudaMalloc(&d_phiReal, arraybytes);
	float* d_thetaReal; cudaMalloc(&d_thetaReal, arraybytes);
	float* d_phiFake; cudaMalloc(&d_phiFake, arraybytes);
	float* d_thetaFake; cudaMalloc(&d_thetaFake, arraybytes);

	// copy data to the GPU
	cudaMemcpy(d_phiReal, h_phiReal, arraybytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_thetaReal, h_thetaReal, arraybytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_phiFake, h_phiFake, arraybytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_thetaFake, h_thetaFake, arraybytes, cudaMemcpyHostToDevice);

	// Size of thread blocks
	int blocksInGrid = (totalPairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	printf("Blocks in grid: %d\n", blocksInGrid);

	// run the kernels on the GPU
	// THIS CALCULATES DR
	// 1. Calculate Real vs. Fake (DR histogram) - already implemented
    printf("================= CALCULATING DR ANGLES =====================\n");
	int *d_histogramDR;
	cudaMalloc(&d_histogramDR, histogrambytes);
	cudaMemset(d_histogramDR, 0, histogrambytes);
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_phiReal, d_thetaReal, d_phiFake, d_thetaFake, d_histogramDR, N);
	memset(h_histogramDR, 0, histogrambytes);
    cudaMemcpy(h_histogramDR, d_histogramDR, histogrambytes, cudaMemcpyDeviceToHost);
	cudaFree(d_histogramDR); 

    // 2. Calculate Real vs. Real (DD histogram)
    printf("================= CALCULATING DD ANGLES =====================\n");
	int *d_histogramDD;
	cudaMalloc(&d_histogramDD, histogrambytes);
	cudaMemset(d_histogramDD, 0, histogrambytes);
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_phiReal, d_thetaReal, d_phiReal, d_thetaReal, d_histogramDD, N);
	memset(h_histogramDD, 0, histogrambytes);
    cudaMemcpy(h_histogramDD, d_histogramDD, histogrambytes, cudaMemcpyDeviceToHost);
	cudaFree(d_histogramDD); 

    // 3. Calculate Fake vs. Fake (RR histogram)
    printf("================= CALCULATING RR ANGLES =====================\n");
	int *d_histogramRR;
	cudaMalloc(&d_histogramRR, histogrambytes);
	cudaMemset(d_histogramRR, 0, histogrambytes);
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_phiFake, d_thetaFake, d_phiFake, d_thetaFake, d_histogramRR, N);
	memset(h_histogramRR, 0, histogrambytes);
    cudaMemcpy(h_histogramRR, d_histogramRR, histogrambytes, cudaMemcpyDeviceToHost);
	cudaFree(d_histogramRR);
	
	printf("DONE!\n");
	
	// Free memory
	cudaFree(d_thetaReal); cudaFree(d_phiReal); 
	cudaFree(d_thetaFake); cudaFree(d_phiFake);
	
	end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("GPU Execution time: %.2f s\n", time_used);
	start = clock();
    
	printf("===================== CALCULATING OMEGA =====================\n");
	// calculate omega values on the CPU
	// Memory management
	size_t omegabytes = DEGREE_RANGE*BINS_PER_DEGREE*sizeof(float);
	float* h_omega = (float*)malloc(omegabytes);
	for (int i=0; i<DEGREE_RANGE*BINS_PER_DEGREE; i++){
		float num = (float)(h_histogramDD[i] - (2*h_histogramDR[i]) + h_histogramRR[i]);
		float den = (float)(h_histogramRR[i]);
		if (den != 0.0f){
			h_omega[i] = (float)(num/den);
		}
	}

	printf("\n");
	printf("================== SUMMARIZING HISTOGRAMS ===================\n");
	printf("Summary for Real vs. Fake (DR):\n");
    verbose_histogram(h_histogramDR);
	printf("\n");
    printf("Summary for Real vs. Real (DD):\n");
    verbose_histogram(h_histogramDD);
	printf("\n");
    printf("Summary for Fake vs. Fake (RR):\n");
    verbose_histogram(h_histogramRR);
	printf("\n");

	// write omega values to omega.out
	printf("=================== SAVING OMEGA VALUES =====================\n");
	std::ofstream outfile("omega.out");
    if (!outfile.is_open()) {
        printf("Error opening file %s for writing.\n", "omega.out");
        return 0;
    }

    for (int i = 0; i < DEGREE_RANGE*BINS_PER_DEGREE; i++) {
        outfile << i << " " << h_omega[i] << "\n";
    }
    outfile.close();
    printf("Omega values saved to %s\n", "omega.out");

	printf("====================== OMEGA SUMMARY ========================\n");

	verbose_omega(h_omega);

	printf("===================== SAVING HISTOGRAMS =====================\n");
	// save the histogram to a file for analyzing later
	save_histogram_to_file(h_histogramDR, DEGREE_RANGE * BINS_PER_DEGREE, "histogramDR.txt");
	save_histogram_to_file(h_histogramDD, DEGREE_RANGE * BINS_PER_DEGREE, "histogramDD.txt");
	save_histogram_to_file(h_histogramRR, DEGREE_RANGE * BINS_PER_DEGREE, "histogramRR.txt");

	printf("======================= DONE, GOODBYE! ======================\n");

	// Free host memory
	free(h_histogramDD); free(h_histogramDR); free(h_histogramRR); free(h_omega);
	end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0f;
	printf("CPU Execution time: %.2f ms\n", time_used);

	return (0);
}

int readdata(char *argv1, char *argv2)
{
	int i, linecount;
	char inbuf[180];
	double ra, dec; // phi, theta, dpi;
	FILE *infil;

	printf("   Assuming input data is given in arc minutes!\n");

	double dpi = 3.14159265;  // TODO: maybe more precision?

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
	h_phiReal = (float *)calloc(nrReal, sizeof(float));
	h_thetaReal = (float *)calloc(nrReal, sizeof(float));

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
		// spherical coordinates phi and theta in radians:
		// phi   = ra/60.0 * dpi/180.0;
		// theta = (90.0-dec/60.0)*dpi/180.0;
		// store values as phi and theta in radians instead of right ascension and declination in arc minutes

		h_phiReal[i] = (float) (ra / 60.0f) * (dpi / 180.0f);
		h_thetaReal[i] = (float) (dec / 60.0f) * (dpi / 180.0f);
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
	h_phiFake = (float *)calloc(nrFake, sizeof(float));
	h_thetaFake = (float *)calloc(nrFake, sizeof(float));

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
		// spherical coordinates phi and theta in radians:
		// phi   = ra/60.0 * dpi/180.0;
		// theta = (90.0-dec/60.0)*dpi/180.0;
		// store values as phi and theta in radians instead of right ascension and declination in arc minutes

		h_phiFake[i] = (float) (ra / 60.0f) * (dpi / 180.0f);
		h_thetaFake[i] = (float) (dec / 60.0f) * (dpi / 180.0f);
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