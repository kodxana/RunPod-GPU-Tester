import json
import os
from urllib import request, parse
import subprocess
import torch
import re  # For parsing nvidia-smi output

def get_pod_machine_id(pod_id, api_key):
    """Retrieve the machineId of a specific pod by its ID using urllib."""
    url = f'https://api.runpod.io/graphql?api_key={api_key}'
    headers = {
        'Content-Type': 'application/json',
    }
    query = """
        query Pod($podId: String!) {
          pod(input: { podId: $podId }) {
            machineId
          }
        }
    """
    data = json.dumps({"query": query, "variables": {"podId": pod_id}}).encode()
    req = request.Request(url, data=data, headers=headers, method='POST')
    try:
        with request.urlopen(req) as response:
            response_body = response.read()
            data = json.loads(response_body)
            return data.get("data", {}).get("pod", {}).get("machineId")
    except Exception as e:
        print(f"Failed to fetch machineId: {e}")
        return None

def collect_env_info():
    print("Collecting environment information...")
    env_info = {
        "RUNPOD_POD_ID": os.getenv("RUNPOD_POD_ID", "Not Available"),
        "Template CUDA_VERSION": os.getenv("CUDA_VERSION", "Not Available"),
        "NVIDIA_DRIVER_CAPABILITIES": os.getenv("NVIDIA_DRIVER_CAPABILITIES", "Not Available"),
        "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES", "Not Available"),
        "NVIDIA_PRODUCT_NAME": os.getenv("NVIDIA_PRODUCT_NAME", "Not Available"),
        "RUNPOD_GPU_COUNT": os.getenv("RUNPOD_GPU_COUNT", "Not Available"),
        "machineId": get_pod_machine_id(os.getenv("RUNPOD_POD_ID"), os.getenv('RUNPOD_API_KEY'))
    }
    return env_info

def parse_nvidia_smi_output(output):
    cuda_version = re.search(r"CUDA Version: (\d+\.\d+)", output)
    driver_version = re.search(r"Driver Version: (\d+\.\d+\.\d+)", output)
    gpu_name = re.search(r"\|\s+\d+\s+([^\|]+?)\s+On\s+\|", output)
    return {
        "CUDA Version": cuda_version.group(1) if cuda_version else "Not Available",
        "Driver Version": driver_version.group(1) if driver_version else "Not Available",
        "GPU Name": gpu_name.group(1).strip() if gpu_name else "Not Available"
    }

def get_nvidia_smi_info():
    try:
        nvidia_smi_output = subprocess.getoutput('nvidia-smi')
        return parse_nvidia_smi_output(nvidia_smi_output)
    except Exception as e:
        return {"Error": f"Failed to fetch nvidia-smi info: {str(e)}"}

def get_system_info():
    system_info = {
        "PyTorch Version": torch.__version__,
        "Environment Info": collect_env_info(),
        "Host Machine Info": get_nvidia_smi_info()
    }
    return system_info

def run_cuda_test():
    print("Performing CUDA operation tests on all available GPUs...")
    gpu_count = int(os.getenv('RUNPOD_GPU_COUNT', torch.cuda.device_count()))
    results = {}

    if gpu_count == 0:
        return {"Error": "No GPUs found."}

    for gpu_id in range(gpu_count):
        try:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            x = torch.rand(10, 10, device=device)
            y = torch.rand(10, 10, device=device)
            z = x + y  # Simple operation to test CUDA
            results[f'GPU {gpu_id}'] = "Success: CUDA is working correctly."
        except Exception as e:
            results[f'GPU {gpu_id}'] = f"Error: {str(e)}"

    return results

def save_info_to_file(info, filename="/workspace/gpu_diagnostics.json"):
    with open(filename, 'w') as file:
        json.dump(info, file, indent=4)
    print(f"Diagnostics information saved to {filename}. Please share this file with RunPod Tech Support for further assistance.")

def main():
    print("RunPod GPU Diagnostics Tool")
    system_info = get_system_info()
    system_info['CUDA Test Result'] = run_cuda_test()
    save_info_to_file(system_info)

if __name__ == "__main__":
    main()
