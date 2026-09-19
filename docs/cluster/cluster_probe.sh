#!/usr/bin/env bash
# Run this on a COMPUTE node of the cluster (srun --pty ... bash), not the login node.
# It answers every question that decides how pinball/ChromScape get installed.
echo "=== arch / os ==="
uname -m; uname -r; cat /etc/os-release 2>/dev/null | head -2
echo "=== cpu / mem ==="
nproc; grep -m1 'model name' /proc/cpuinfo 2>/dev/null || lscpu | grep -m1 'Model name'
free -g | head -2
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
echo "=== cuda toolkit ==="
which nvcc && nvcc --version | tail -2
echo "=== schedulers / containers ==="
for c in sbatch srun apptainer singularity podman docker module; do
  printf '%-12s %s\n' "$c" "$(command -v $c || echo '-')"
done
echo "=== module avail (cuda/python/conda) ==="
(module avail 2>&1 | grep -iE 'cuda|python|conda|anaconda|mamba' | head -20) || echo "no modules"
echo "=== filesystems (where do big things live) ==="
df -h "$HOME" /scratch /work /lustre 2>/dev/null | sort -u
echo "=== quota ==="
quota -s 2>/dev/null | head -5 || echo "no quota cmd"
echo "=== outbound net (needed for pip/conda/HF) ==="
timeout 8 curl -sSI https://pypi.org/simple/ | head -1 || echo "PyPI UNREACHABLE -> need an offline/mirror plan"
timeout 8 curl -sSI https://github.com | head -1 || echo "GitHub UNREACHABLE"
