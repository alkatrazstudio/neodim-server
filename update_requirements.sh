#!/usr/bin/env bash
set -e
cd "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR=.venv

rm -rf "$VENV_DIR"
if [[ -f requirements.txt ]]
then
    mv requirements.txt "requirements.txt.$(date "+%Y%m%d_%H%M%S").bak"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

set -a
source pip.env
set +a

pip install -r requirements_direct.txt
echo "--extra-index-url https://download.pytorch.org/whl/cu117" > requirements.txt
pip freeze >> requirements.txt

REQ_HASH=($(cat pip.env requirements.txt | md5sum))
printf "%s" "${REQ_HASH[0]}" > .requirements_hash
