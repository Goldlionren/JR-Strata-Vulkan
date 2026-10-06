#!/usr/bin/env bash
set -euo pipefail

# V9 pre-start history rotation + plan refresh.
#
# Run this before starting Strata.  It:
#   1. archives the previous run's routing trace, if present;
#   2. rebuilds the rolling Strata profile + Vulkan residency plan;
#   3. leaves a clean path for the next --dump-routing run.

VULKAN_ROOT="${VULKAN_ROOT:-/data/strata-lab/JR-Strata-Vulkan}"
STRATA_DATA="${STRATA_DATA:-/data/strata-lab/Strata/data}"
CURRENT_TRACE="${CURRENT_TRACE:-$STRATA_DATA/v9-production-routing.bin}"
HISTORY_DIR="${HISTORY_DIR:-$STRATA_DATA/routing-history}"

mkdir -p "$HISTORY_DIR" "$VULKAN_ROOT/data"

if [[ -s "$CURRENT_TRACE" ]]; then
    ts="$(date -r "$CURRENT_TRACE" +%Y%m%d-%H%M%S)"
    dst="$HISTORY_DIR/v9-routing-$ts.bin"
    # Avoid accidental overwrite if two traces have the same mtime second.
    n=1
    while [[ -e "$dst" ]]; do
        dst="$HISTORY_DIR/v9-routing-$ts-$n.bin"
        n=$((n + 1))
    done
    mv "$CURRENT_TRACE" "$dst"
    echo "[V9] archived: $dst"
elif [[ -e "$CURRENT_TRACE" ]]; then
    rm -f "$CURRENT_TRACE"
fi

python3 "$VULKAN_ROOT/tools/vulkan_v9_history_update.py" "$@"

echo
echo "[V9] next Strata run should keep:"
echo "     --dump-routing /work/Strata/data/v9-production-routing.bin"
echo
echo "[V9] for SYCL Strata, point --expert-profile once at:"
echo "     /work/Strata/data/expert-profile-v9-history.bin"
echo
echo "[V9] Vulkan plan path stays stable:"
echo "     $VULKAN_ROOT/data/v9-cache-plan-history.bin"
