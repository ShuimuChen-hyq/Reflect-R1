#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

require_var SIGLIP_MODEL_PATH

FLOW_CONFIG="${CACHE_DIR}/torch-flow.runtime.yml"
cat > "${FLOW_CONFIG}" <<YAML
jtype: Flow
version: '1'
with:
  port: ${CLIP_PORT}
executors:
  - name: clip_t
    uses:
      jtype: CLIPEncoder
      with:
        name: ${SIGLIP_MODEL_PATH}
        device: ${SIGLIP_DEVICE:-cuda}
      metas:
        py_modules:
          - clip_server.executors.clip_torch
    timeout_ready: ${CLIP_TIMEOUT_READY:-3000000}
    replicas: ${CLIP_REPLICAS:-1}
YAML

echo "SIGLIP_URL=${SIGLIP_URL}"
echo "Flow config: ${FLOW_CONFIG}"
exec "${PYTHON}" -m clip_server "${FLOW_CONFIG}"
