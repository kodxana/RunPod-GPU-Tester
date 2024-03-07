
# RunPod GPU Tester

The `RunPod-GPU-Tester` is a diagnostic tool designed to help users and tech support diagnose and troubleshoot GPU-related issues in RunPod environments. This Python script collects essential system and environment information, performs basic CUDA operations to test each GPU's functionality, and generates a diagnostic report that users can share with RunPod tech support for further assistance.

## Features

- Collects system and GPU environment information.
- Performs CUDA operations to test the functionality of all available GPUs.
- Generates a detailed diagnostic report (`gpu_diagnostics.json`) including any errors encountered during the tests.

## Prerequisites

- Python 3.6 or later.
- Access to a RunPod environment with GPUs (obviusly) .

## Quick Start

To use the `RunPod-GPU-Tester`, follow these simple steps:

1. Download the script directly into your RunPod environment using the following command:

   ```
   wget https://github.com/kodxana/RunPod-GPU-Tester/raw/main/tester.py -O tester.py
   ```

2. Run the script using Python:

   ```
   python tester.py
   ```

3. After the script completes, it will generate a `gpu_diagnostics.json` file in the `/workspace` directory. Share this file with RunPod tech support for further analysis and assistance.

## Output

The script will save the diagnostic information in a file named `gpu_diagnostics.json` in the `/workspace` directory. This file contains:

- System information including CUDA version, PyTorch version, and GPU details.
- Results from the CUDA functionality tests for each GPU.
- The machine ID associated with the current RunPod pod .

## Contributing

Contributions to the `RunPod-GPU-Tester` are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## Support

If you encounter any issues or have questions about using the script, please open an issue in this GitHub repository.
