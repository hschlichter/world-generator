#!/bin/sh
# usdview with the project's OpenUSD build and the UsdLod plugin (menu "UsdLod").
#   generator/usdview.sh [world/root.usda] [usdview options, e.g. --unloaded]
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export PYTHONPATH="$ROOT/openusd_build/lib/python3.14/site-packages:$HERE:$HERE/usdview_plugin${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/openusd_build/bin:$PATH"
export PXR_PLUGINPATH_NAME="$HERE/usdview_plugin/streamworldUsdview${PXR_PLUGINPATH_NAME:+:$PXR_PLUGINPATH_NAME}"
FILE="${1:-$ROOT/world/root.usda}"
[ $# -gt 0 ] && shift
exec usdview "$FILE" "$@"
