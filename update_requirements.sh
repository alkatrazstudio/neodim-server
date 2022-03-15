#!/usr/bin/env bash
set -e
cd "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR=.venv

rm -rf "$VENV_DIR"
mv requirements.txt "requirements.txt.$(date "+%Y%m%d_%H%M%S").bak"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -r requirements_direct.txt
pip freeze > requirements.txt
