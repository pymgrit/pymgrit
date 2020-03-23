#!/usr/bin/env bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    apt update
    apt install -y mpich
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install mpich
fi

pip3 install .
pip3 install sphinx sphinx-rtd-theme
sphinx-apidoc -o docs/source src/pymgrit
sphinx-build -b html docs/source docs/build/html

touch docs/build/html/.nojekyll
