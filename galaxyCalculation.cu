#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>  // For std::sort
#include <vector>     // For std::vector and std::pair
#include <limits.h>
#include <fstream>  // Include for file output

#define BIN_RANGE 180
#define BINS_PER_DEGREE 4
#define THREADS_PER_BLOCK 512
#define ARCMIN_TO_RAD 0.0029088820866
#define RAD_TO_DEG 57.29577951

__global__ void calculateAngle(float* ra_A, float* decl_A, float* ra_B, float* decl_B, unsigned int* histogram, int N, bool differentSet){
	float theta_deg;
	
	long long int r = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
	int i = r / N;  // index of A 
	int k = r % N;  // index of B

	if (i >= N || k >= N) {  // check that we're in bounds
		return;
	}

	if (!differentSet && i >= k){  // Only process unique pairs where i < k
		// The pair (i, k) and (k, i) is the same and has been accounted for twice
		return;
	}

	// convert angles from arc minutes to radians

	float alpha_A = ra_A[i] * ARCMIN_TO_RAD;
	float delta_A = decl_A[i] * ARCMIN_TO_RAD; 
	float alpha_B = ra_B[k] * ARCMIN_TO_RAD;
	float delta_B = decl_B[k] * ARCMIN_TO_RAD;

	// TODO: check that we have logical values

	float dotProduct = cos(delta_A)*cos(delta_B)*cos(alpha_A-alpha_B)+sin(delta_A)*sin(delta_B);
	dotProduct = fminf(1.0f, fmaxf(dotProduct, -1.0f)); // Clamp to [-1, 1]

	float theta_rad = acosf(dotProduct);
	theta_deg = theta_rad * RAD_TO_DEG;

	// histogram time!
	int histogramBinIndex = floor(theta_deg * BINS_PER_DEGREE);
	if (histogramBinIndex >= 0 && histogramBinIndex < BIN_RANGE * BINS_PER_DEGREE){ // ensure boundary
		atomicInc(&histogram[histogramBinIndex], UINT_MAX);  // this probably slows down the execution time a lot
		if (!differentSet){
			// We need to increment again if the sets are the same because we're halving the comparisons
			atomicInc(&histogram[histogramBinIndex], UINT_MAX);
		}
	}
}

void summarize_histogram(unsigned int* histogram){
	long total_count = 0;
	int NUM_BINS = BIN_RANGE * BINS_PER_DEGREE;

	// Calculate total count, find max height and mode, and locate non-zero range
    for (int i = 0; i < NUM_BINS; i++) {
        total_count += histogram[i];
    }
    
    // Calculate mean bin height
    double mean_height = (double)total_count / NUM_BINS;
    
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
}

void verbose_histogram(unsigned int* histogram) {
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
    
    // Mean and standard deviation
    double mean_height = (double)total_count / NUM_BINS;
    double variance = 0.0;
    for (int i = 0; i < NUM_BINS; i++) {
        variance += pow(histogram[i] - mean_height, 2);
    }
    variance /= NUM_BINS;
    double stddev = sqrt(variance);

    // Prepare vector of (bin index, count) pairs for sorting
    std::vector<std::pair<int, unsigned int>> bin_counts;
    for (int i = 0; i < NUM_BINS; i++) {
        bin_counts.push_back(std::make_pair(i, histogram[i]));
    }

    // Sort by count in descending order for top 3, ascending for bottom 3
    std::sort(bin_counts.begin(), bin_counts.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    // Top 3 bins with the most entries
    printf("\nTop 3 most populated bins:\n");
    for (int j = 0; j < 3 && j < bin_counts.size(); j++) {
        int bin_idx = bin_counts[j].first;
        unsigned int count = bin_counts[j].second;
        float angle = (float)bin_idx / BINS_PER_DEGREE;  // Convert bin index to angle in degrees
        printf("Bin %d (Angle ≈ %.2f°): Count = %d\n", bin_idx, angle, count);
    }

    // Sort by ascending order to find bottom 3 non-zero populated bins
    std::sort(bin_counts.begin(), bin_counts.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

    // Bottom 3 bins with the least entries (excluding zero counts)
    printf("\nBottom 3 least populated bins (non-zero):\n");
    int count_bottom = 0;
    for (const auto& bin : bin_counts) {
        if (bin.second > 0) {  // Exclude zero entries
            int bin_idx = bin.first;
            unsigned int count = bin.second;
            float angle = (float)bin_idx / BINS_PER_DEGREE;  // Convert bin index to angle in degrees
            printf("Bin %d (Angle ≈ %.2f°): Count = %d\n", bin_idx, angle, count);
            count_bottom++;
            if (count_bottom >= 3) break;  // Only get bottom 3
        }
    }

    // Output summary
    printf("\nHistogram Summary:\n");
    printf("Total count: %ld\n", total_count);
    printf("Mean bin height: %.2f\n", mean_height);
    printf("Mode bin: %d with height %d\n", mode_bin, max_height);
    printf("Standard deviation of bin heights: %.2f\n", stddev);
    printf("Non-zero bin range: %d to %d\n", non_zero_start, non_zero_end);
}

void save_histogram_to_file(const unsigned int* histogram, int num_bins, const char* filename) {
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
	printf("==================== READING INPUT DATA =====================\n");
	int readdata(char *argv1, char *argv2);
	printf("==================== READING DEVICE DATA ====================\n");
	int getDevice(int deviceNr);
	// long int histogramDRsum, histogramDDsum, histogramRRsum;
	// struct timeval _ttime;
	// struct timezone _tzone;
	// cudaError_t myError;

	FILE *outfil;

	printf("================= CALCULATING DR ANGLES =====================\n");

	if (argc != 4)
	{
		printf("Usage: a.out real_data random_data output_data\n");
		return (-1);
	}

	if (getDevice(0) != 0)
		return (-1);

	if (readdata(argv[1], argv[2]) != 0)
		return (-1);

	// make sure input array sizes are the same
	
	int N = nrReal;
	if (N != nrFake){
		printf("Input data lengths are not the same!");
		return (-1);
	}

	clock_t start, end;
	double time_used;
	start = clock();

	long long totalPairs = (long long)N*N;

	// allocate memory on the GPU and histogram arrays memory on CPU
	size_t arraybytes = N * sizeof(float);
	printf("Arraybytes: %d\n", arraybytes);

	size_t histogrambytes = BIN_RANGE * BINS_PER_DEGREE * sizeof(unsigned int);
	unsigned int* h_histogramDR = (unsigned int*)malloc(histogrambytes);
	unsigned int* h_histogramDD = (unsigned int*)malloc(histogrambytes);
	unsigned int* h_histogramRR = (unsigned int*)malloc(histogrambytes);
	
	unsigned int *d_histogramDR, *d_histogramDD, *d_histogramRR;
	cudaMalloc(&d_histogramDR, histogrambytes);
	cudaMalloc(&d_histogramDD, histogrambytes);
	cudaMalloc(&d_histogramRR, histogrambytes);

	// allocate to GPU: the real and fake right ascension and declination
	float* d_raReal; cudaMalloc(&d_raReal, arraybytes);
	float* d_declReal; cudaMalloc(&d_declReal, arraybytes);
	float* d_raFake; cudaMalloc(&d_raFake, arraybytes);
	float* d_declFake; cudaMalloc(&d_declFake, arraybytes);

	// copy data to the GPU
	cudaMemcpy(d_raFake, h_raFake, arraybytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_declFake, h_declFake, arraybytes, cudaMemcpyHostToDevice);

	// Size of thread blocks
	int blocksInGrid = (totalPairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	printf("Blocks in grid: %d\n", blocksInGrid);

	// run the kernels on the GPU
	// THIS CALCULATES DR
	// 1. Calculate Real vs. Fake (DR histogram) - already implemented
    printf("================= CALCULATING DR ANGLES =====================\n");
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_raReal, d_declReal, d_raFake, d_declFake, d_histogramDR, N, true);
    cudaMemcpy(h_histogramDR, d_histogramDR, histogrambytes, cudaMemcpyDeviceToHost);

    // 2. Calculate Real vs. Real (DD histogram)
    printf("================= CALCULATING DD ANGLES =====================\n");
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_raReal, d_declReal, d_raReal, d_declReal, d_histogramDD, N, false);
    cudaMemcpy(h_histogramDD, d_histogramDD, histogrambytes, cudaMemcpyDeviceToHost);
	h_histogramDD[0] += N;  // N galaxies has 0 angle with itself, so add N to the first bin

    // 3. Calculate Fake vs. Fake (RR histogram)
    printf("================= CALCULATING RR ANGLES =====================\n");
    calculateAngle<<<blocksInGrid, THREADS_PER_BLOCK>>>(d_raFake, d_declFake, d_raFake, d_declFake, d_histogramRR, N, false);
    cudaMemcpy(h_histogramRR, d_histogramRR, histogrambytes, cudaMemcpyDeviceToHost);
	h_histogramRR[0] += N;
	
	printf("DONE!\n");
	
	// Free memory
	cudaFree(d_declReal); cudaFree(d_raReal); 
	cudaFree(d_declFake); cudaFree(d_raFake);
	cudaFree(d_histogramDR); cudaFree(d_histogramDD); cudaFree(d_histogramRR);

	end = clock();
	time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
	printf("Execution time: %.2f ms\n", time_used);
	

	printf("=================== SUMMARIZING HISTOGRAM ===================\n");
	printf("Summary for Real vs. Fake (DR):\n");
    verbose_histogram(h_histogramDR);

    printf("Summary for Real vs. Real (DD):\n");
    verbose_histogram(h_histogramDD);

    printf("Summary for Fake vs. Fake (RR):\n");
    verbose_histogram(h_histogramRR);

    // Free host memory
    free(h_histogramDR); free(h_histogramDD); free(h_histogramRR);
	
	// calculate omega values on the CPU, can of course be done on the GPU

	printf("===================== SAVING HISTOGRAMS =====================\n");
	// After each call to `verbose_histogram`, save the histogram to a file
	save_histogram_to_file(h_histogramDR, BIN_RANGE * BINS_PER_DEGREE, "histogramDR.txt");
	save_histogram_to_file(h_histogramDD, BIN_RANGE * BINS_PER_DEGREE, "histogramDD.txt");
	save_histogram_to_file(h_histogramRR, BIN_RANGE * BINS_PER_DEGREE, "histogramRR.txt");


	printf("======================= DONE, GOODBYE! ======================\n");
	return (0);
}

int readdata(char *argv1, char *argv2)
{
	int i, linecount;
	char inbuf[180];
	double ra, dec; // phi, theta, dpi;
	FILE *infil;

	printf("   Assuming input data is given in arc minutes!\n");
	// spherical coordinates phi and theta in radians:
	// phi   = ra/60.0 * dpi/180.0;
	// theta = (90.0-dec/60.0)*dpi/180.0;

	// dpi = acos(-1.0);
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
