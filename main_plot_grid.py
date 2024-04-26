#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir

from plots.grid import grayscale



path_in = "outputs/parameter_eval/"
path_out = "outputs/parameter_plot/"

for fn in listdir(path_in):
    filename = fn.split(".")[0]
    if len(fn.split(".")) > 1 and fn.split(".")[1] == "json":
        grayscale(path_in + filename + ".json", save_file = filename + ".png", save_dir = path_out)
