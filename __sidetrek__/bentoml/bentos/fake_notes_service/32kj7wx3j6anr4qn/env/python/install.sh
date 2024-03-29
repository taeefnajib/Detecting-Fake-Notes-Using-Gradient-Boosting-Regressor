#!/usr/bin/env bash
set -exuo pipefail

# Parent directory https://stackoverflow.com/a/246128/8643197
BASEDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"

PIP_ARGS=(--no-warn-script-location)

# BentoML by default generates two requirement files:
#  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`
#  - ./env/python/requirements.txt: all dependencies as user specified in code or requirements.txt file
REQUIREMENTS_TXT="$BASEDIR/requirements.txt"
REQUIREMENTS_LOCK="$BASEDIR/requirements.lock.txt"
WHEELS_DIR="$BASEDIR/wheels"
BENTOML_VERSION=${BENTOML_VERSION:-1.0.15}
# Install python packages, prefer installing the requirements.lock.txt file if it exist
if [ -f "$REQUIREMENTS_LOCK" ]; then
    echo "Installing pip packages from 'requirements.lock.txt'.."
    pip3 install -r "$REQUIREMENTS_LOCK" "${PIP_ARGS[@]}"
else
    if [ -f "$REQUIREMENTS_TXT" ]; then
        echo "Installing pip packages from 'requirements.txt'.."
        pip3 install -r "$REQUIREMENTS_TXT" "${PIP_ARGS[@]}"
    fi
fi

# Install user-provided wheels
if [ -d "$WHEELS_DIR" ]; then
    echo "Installing wheels packaged in Bento.."
    pip3 install "$WHEELS_DIR"/*.whl "${PIP_ARGS[@]}"
fi

# Install the BentoML from PyPI if it's not already installed
if python3 -c "import bentoml" &> /dev/null; then
    existing_bentoml_version=$(python3 -c "import bentoml; print(bentoml.__version__)")
    if [ "$existing_bentoml_version" != "$BENTOML_VERSION" ]; then
        echo "WARNING: using BentoML version ${existing_bentoml_version}"
    fi
else
    pip3 install bentoml=="$BENTOML_VERSION"
fi
