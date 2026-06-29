#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

run_grpo_entry "v11_valid_tool_split_S123_no_reasoning" "$@"
