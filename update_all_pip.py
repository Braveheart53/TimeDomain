# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import pip
from subprocess import call
from pip._internal.utils.misc import get_installed_distributions

packages = [dist.project_name for dist in get_installed_distributions()]
call("python -m pip install --user --upgrade " + ' '.join(packages), shell=True)
