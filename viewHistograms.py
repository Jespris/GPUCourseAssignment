
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants for bin grouping
BINS_PER_DEGREE = 4  # 4 bins per degree
ANGLE_RANGE = 90  # Total angle range in degrees


def get_histogram_values(data):
    bin_counts = data[:, 1]  # Histogram counts for each bin

    # Generate angle ranges for the bins
    angles = np.arange(0, ANGLE_RANGE*BINS_PER_DEGREE)

    return np.array(angles), np.array(bin_counts)


def plot_individual_histogram(file_path, output_image):
    """
    Plots and saves an individual histogram with the specified formatting.
    """
    data = np.loadtxt(file_path, delimiter=' ')
    angles, angle_counts = get_histogram_values(data)

    # Plot with log scale on y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(angles, angle_counts, color='blue')
    plt.yscale('log')
    plt.xlabel("Bin")
    plt.ylabel("Count (log scale)")
    plt.title(f"Histogram for {os.path.basename(output_image)}")
    plt.grid(True, which="both", ls="--", lw=0.5)

    # Save the individual plot as an image
    plt.savefig(output_image)
    plt.close()
    print(f"Saved individual plot to {output_image}")


def plot_omega(file_path, output_image):
    data = np.loadtxt(file_path, delimiter=' ')
    omega_values = np.array(data[:, 0])
    omega_counts = np.array(data[:, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(omega_values, omega_counts, color='red')
    plt.xlabel("Bin index")
    plt.ylabel("Omega")
    plt.title(f"Omega gram")
    plt.grid(True)

    plt.savefig(output_image)
    plt.close()
    print(f"Saved omega gram to {output_image}")


def plot_combined_histogram(histogram_files, output_dir):
    """
    Plots all histograms on the same graph with the specified angle aggregation and formatting.
    """
    plt.figure(figsize=(10, 6))

    # Define colors for each histogram
    colors = {"DR": "blue", "DD": "green", "RR": "red"}

    for label, (file_path, output_image) in histogram_files.items():
        data = np.loadtxt(file_path, delimiter=' ')
        angles, angle_counts = get_histogram_values(data)

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

    plot_omega("output/omega.out", f"{output_dir}/omegaGram.png")
