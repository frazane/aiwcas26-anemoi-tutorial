#!/bin/bash
#SBATCH --job-name=anemoi-smoke
#SBATCH --account=tra26_aiwcas
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

# Minimal check that the anemoi environment sees a GPU. Usage:
#   ANEMOI_VENV=/path/to/venv sbatch scripts/smoke_test_gpu.sbatch
set -euo pipefail

module load python/3.11.7
source "${ANEMOI_VENV:-$SCRATCH/envs/anemoi}/bin/activate"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python - <<'PY'
import torch
import torch_geometric
import anemoi.datasets, anemoi.graphs, anemoi.models, anemoi.training

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible"
print("device", torch.cuda.get_device_name(0))

x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", float((x @ x).sum().isfinite()))

from torch_geometric.nn import GATConv
conv = GATConv(16, 16).cuda()
edge_index = torch.randint(0, 1000, (2, 5000), device="cuda")
print("pyg GATConv ok:", tuple(conv(torch.randn(1000, 16, device="cuda"), edge_index).shape))

for m in (anemoi.datasets, anemoi.graphs, anemoi.models, anemoi.training):
    print(m.__name__, m.__version__)
print("SMOKE TEST PASSED")
PY
