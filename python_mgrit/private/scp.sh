#!/usr/bin/env bash
rsync -avh --exclude 'private' --exclude 'results' --exclude 'PyMgrit.egg-info' /home/jens/uni/pasirom/python_mgrit hahne@wmwrsrv2.math.uni-wuppertal.de:/home/hahne
