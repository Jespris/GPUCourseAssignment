# Define variables
$runRemotely = $true  # Change this to $false to run locally
$remoteUser = "jespande"
$remoteHost = "dione.utu.fi"
$localOutputDir = "./output"
$remoteFiles = @("err.txt", "out.txt", "omega.txt", "histogramDD.txt", "histogramDR.txt", "histogramRR.txt")

if ($runRemotely){
    # Prompt for password securely
    Write-Output "Enter your password:"
    $passwordSecure = Read-Host -AsSecureString
    $password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($passwordSecure))


    # 1. Copy galaxyCalculation.cu to the remote cluster
    Write-Output "Copying cuda program to the remote cluster..."
    & pscp -pw $password gpuUppg.cu "$remoteUser@${remoteHost}:"

    # 2. SSH to the remote cluster and run commands
    Write-Output "Running commands on the remote cluster..."
    & plink -pw $password $remoteUser@$remoteHost "module load cuda && module load GCC/7.3.0-2.30 && nvcc -O3 -arch=sm_70 -o galaxy gpuUppg.cu && srun -p gpu -n 1 -t 10:00 --mem=10G -e err.txt -o out.txt ./galaxy data_100k_arcmin.dat flat_100k_arcmin.dat omega.out && echo 'Job completed on remote cluster.'"

    # 3. Copy output files from the remote cluster to the local machine
    Write-Output "Copying result files from the remote cluster..."
    New-Item -ItemType Directory -Path $localOutputDir -Force | Out-Null  # Ensure local output directory exists

    foreach ($file in $remoteFiles) {
        & pscp -pw $password "$remoteUser@${remoteHost}:$file" "$localOutputDir/"
    }

    Write-Output "All files copied successfully to $localOutputDir."
} else {
    # local execution
    # Ensure CUDA is available locally
    if (-Not (Get-Command "nvcc" -ErrorAction SilentlyContinue)) {
        Write-Output "nvcc is not available on the local machine."
        exit
    }

    # Compile galaxyCalculation.cu locally
    Write-Output "Compiling cuda program locally..."
    & nvcc -O3 -arch=sm_70 -o galaxy galaxyCalculation.cu

    # Execute the compiled binary locally
    Write-Output "Executing galaxy locally..."
    & ./galaxy data_100k_arcmin.dat flat_100k_arcmin.dat omega.out > out.txt 2> err.txt

    # Move results to the output directory
    New-Item -ItemType Directory -Path $localOutputDir -Force | Out-Null  # Ensure local output directory exists
    Move-Item -Path "out.txt" -Destination "$localOutputDir/"
    Move-Item -Path "err.txt" -Destination "$localOutputDir/"
    Move-Item -Path "omega.out" -Destination "$localOutputDir/"
}

# 4. Ensure required Python packages are installed
Write-Output "Ensuring required Python packages are installed..."
python -m pip install --user matplotlib numpy

# 5. Run the Python script viewHistograms.py
Write-Output "Running viewHistograms.py..."
python viewHistograms.py

Write-Output "Python script executed successfully."