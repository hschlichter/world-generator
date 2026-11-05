# Source this to use the project's OpenUSD build and generator package:  . generator/env.sh
_SW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export PYTHONPATH="$_SW_ROOT/openusd_build/lib/python3.14/site-packages:$_SW_ROOT/generator"
export PATH="$_SW_ROOT/openusd_build/bin:$PATH"
