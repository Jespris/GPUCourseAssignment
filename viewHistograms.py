
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants for bin grouping
BINS_PER_DEGREE = 4  # 4 bins per degree
DEGREE_AGGREGATION = 2  # Aggregate over 2 degrees
BINS_PER_AGGREGATED_DEGREE = BINS_PER_DEGREE * DEGREE_AGGREGATION  # Total bins per 2 degrees
ANGLE_RANGE = 180  # Total angle range in degrees


def aggregate_histogram(data, bins_per_aggregated_degree):
    """
    Aggregates the histogram counts over specified bin intervals to reduce noise.
    """
    bin_counts = data[:, 1]  # Histogram counts for each bin
    aggregated_counts = []

    for i in range(0, len(bin_counts), bins_per_aggregated_degree):
        # Sum counts for every group of bins representing DEGREE_AGGREGATION degrees
        aggregated_counts.append(np.sum(bin_counts[i:i + bins_per_aggregated_degree]))

    # Generate angle ranges for the aggregated bins (e.g., 0, 2, 4, ...)
    angles = np.arange(0, ANGLE_RANGE, DEGREE_AGGREGATION)

    return np.array(angles), np.array(aggregated_counts)


def plot_individual_histogram(file_path, output_image):
    """
    Plots and saves an individual histogram with the specified angle aggregation and formatting.
    """
    data = np.loadtxt(file_path, delimiter=' ')
    angles, angle_counts = aggregate_histogram(data, BINS_PER_AGGREGATED_DEGREE)

    # Plot with log scale on y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(angles, angle_counts, color='blue')
    plt.yscale('log')
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Count (log scale)")
    plt.title(f"Histogram for {os.path.basename(output_image)}")
    plt.grid(True, which="both", ls="--", lw=0.5)

    # Save the individual plot as an image
    plt.savefig(output_image)
    plt.close()
    print(f"Saved individual plot to {output_image}")


def plot_combined_histogram(histogram_files, output_dir):
    """
    Plots all histograms on the same graph with the specified angle aggregation and formatting.
    """
    plt.figure(figsize=(10, 6))

    # Define colors for each histogram
    colors = {"DR": "blue", "DD": "green", "RR": "red"}

    for label, (file_path, output_image) in histogram_files.items():
        data = np.loadtxt(file_path, delimiter=' ')
        angles, angle_counts = aggregate_histogram(data, BINS_PER_AGGREGATED_DEGREE)

        # Plot each histogram line with a unique color and label
        plt.plot(angles, angle_counts, color=colors[label], label=label)

    # Log scale for y-axis and additional grid lines
    plt.yscale('log')
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Count (log scale)")
    plt.title("Combined Histogram for DR, DD, RR")
    plt.grid(True, which="both", ls="--", lw=0.5)  # More support lines
    plt.legend()  # Add legend to identify each histogram

    combined_path = os.path.join(output_dir, "combined_histogram.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved combined histogram plot as {combined_path}")


if __name__ == "__main__":
    output_dir = "histograms"
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary of histogram file names and output images
    histogram_files = {
        "DR": ("output/histogramDR.txt", f"{output_dir}/histogramDR.png"),
        "DD": ("output/histogramDD.txt", f"{output_dir}/histogramDD.png"),
        "RR": ("output/histogramRR.txt", f"{output_dir}/histogramRR.png")
    }

    # Plot each histogram individually
    for label, (file_path, output_image) in histogram_files.items():
        plot_individual_histogram(file_path, output_image)

    # Plot all histograms together in a combined plot
    plot_combined_histogram(histogram_files, output_dir)
