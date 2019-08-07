#!/usr/bin/env bash
rsync -avh --exclude 'private' --exclude '__pycache__' --exclude 'results' --exclude 'PyMgrit.egg-info' /home/jens/uni/pymgrit hahne@wmwrsrv2.math.uni-wuppertal.de:/home/hahne
