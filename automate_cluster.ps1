# Define variables
$remoteUser = "jespande"
$remoteHost = "dione.utu.fi"
$localOutputDir = "./output"
$remoteFiles = @("err.txt", "out.txt", "omega.out", "histogramDD.txt", "histogramDR.txt", "histogramRR.txt")

# Prompt for password securely
Write-Output "Enter your password:"
$passwordSecure = Read-Host -AsSecureString
$password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($passwordSecure))


# 1. Copy galaxyCalculation.cu to the remote cluster
Write-Output "Copying galaxyCalculation.cu to the remote cluster..."
& pscp -pw $password galaxyCalculation.cu "$remoteUser@${remoteHost}:"

# 2. SSH to the remote cluster and run commands
Write-Output "Running commands on the remote cluster..."
& plink -pw $password $remoteUser@$remoteHost "module load cuda && module load GCC/7.3.0-2.30 && nvcc -O3 -arch=sm_70 -o galaxy galaxyCalculation.cu && srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt ./galaxy data_100k_arcmin.dat flat_100k_arcmin.dat omega.out && echo 'Job completed on remote cluster.'"

# 3. Copy output files from the remote cluster to the local machine
Write-Output "Copying result files from the remote cluster..."
New-Item -ItemType Directory -Path $localOutputDir -Force | Out-Null  # Ensure local output directory exists

foreach ($file in $remoteFiles) {
    & pscp -pw $password "$remoteUser@${remoteHost}:$file" "$localOutputDir/"
}

Write-Output "All files copied successfully to $localOutputDir."

# 4. Ensure required Python packages are installed
Write-Output "Ensuring required Python packages are installed..."
python -m pip install --user matplotlib numpy

# 5. Run the Python script viewHistograms.py
Write-Output "Running viewHistograms.py..."
python viewHistograms.py

Write-Output "Python script executed successfully."