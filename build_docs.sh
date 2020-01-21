#!/usr/bin/env bash

sudo apt update
sudo apt install -y mpich

pip install .
pip install sphinx
sphinx-apidoc -o docs/source src/pymgrit
sphinx-build -b html docs/source docs/build/html
