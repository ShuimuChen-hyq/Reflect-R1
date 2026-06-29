#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ -z "${OMPI_COMM_WORLD_RANK:-}" || -z "${OMPI_COMM_WORLD_LOCAL_RANK:-}" || -z "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
  echo "_sft_mpi_worker.sh must be launched by mpirun." >&2
  exit 1
fi

export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}

build_sft_python_command
exec "${SFT_PYTHON_CMD[@]}" "$@"
