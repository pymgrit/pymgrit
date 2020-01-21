#!/usr/bin/env bash

apt update
apt install -y mpich

pip install .
pip install sphinx
sphinx-apidoc -o docs/source src/pymgrit
sphinx-build -b html docs/source docs/build/html

touch docs/build/html/.nojekyll
