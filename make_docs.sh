#!/usr/bin/env bash

sphinx-apidoc -o docs/source src/pymgrit
sphinx-build -b html docs/source docs/build/html
