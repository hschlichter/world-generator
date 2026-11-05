#!/bin/sh
# Runs usdview headless (testusdview, Qt offscreen) on the root stage with the UsdLod plugin.
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
export PYTHONPATH="$ROOT/openusd_build/lib/python3.14/site-packages:$HERE:$HERE/usdview_plugin${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/openusd_build/bin:$PATH"
export PXR_PLUGINPATH_NAME="$HERE/usdview_plugin/streamworldUsdview${PXR_PLUGINPATH_NAME:+:$PXR_PLUGINPATH_NAME}"
export SW_USDVIEW_OUT="${SW_USDVIEW_OUT:-$ROOT/docs/previews}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
mkdir -p "$SW_USDVIEW_OUT"
exec testusdview --testScript "$HERE/usdview_plugin/usdview_check.py" --unloaded "$ROOT/world/root.usda"
