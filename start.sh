#!/usr/bin/env bash
set -e
pushd "$(dirname -- "${BASH_SOURCE[0]}")" > /dev/null
ROOT_DIR="$(pwd)"

VENV_DIR=.venv
HASH_FILE=.requirements_hash


function requirements_hash() {
    A=($(cat pip.env requirements.txt | md5sum))
    printf "%s" "${A[0]}"
}

function reinstall() {
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    set -a
    source pip.env
    set +a

    pip install -r requirements.txt
    requirements_hash > "$HASH_FILE"
}

function init_env() {
    if [[ -d "$VENV_DIR" && -f "$HASH_FILE" ]]
    then
        SAVED_HASH="$(< "$HASH_FILE")"
        RELEASE_HASH="$(requirements_hash)"
        if [[ "$SAVED_HASH" == "$RELEASE_HASH" ]]
        then
            source "$VENV_DIR/bin/activate"
            return
        fi
    fi

    reinstall
    echo
    echo "------------------------------------------"
    echo
}

init_env

popd > /dev/null

export BITSANDBYTES_NOWELCOME=1
export PYTHONWARNINGS="ignore:TypedStorage is deprecated"
python "$ROOT_DIR/src/main.py" "$@"
