# -*- coding: utf-8 -*-
# =============================================================================
# """
# Created on Fri Sep 27 17:15:36 2019
# William W. Wallace
# @author: V448789
# """
# =============================================================================

# =============================================================================
# """
# Log:
#     
#     Need to create a function here that returns when calling
#     W3_unit_conver.function() so that I do need to copy this for all
#     programs using in the future
# """
# =============================================================================
from IPython import get_ipython;   
get_ipython().magic('reset -sf')

# Debugging
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


# Load required libraries / package'
# import cmath as cma
#import numpy as np
import scipy.constants as cons
import mpmath as mpm
# using my own conversion factors, but for an expanded list, I could use pint
#import pint as pint
#ureg = pint.UnitRegistry()

# constants
c_0 = cons.speed_of_light


# User Input Required
Range_length = 14.5   
Tx_ht = 6
Rcv_ht = 6
Fresnel_Zones = 6
units_O_len = 'ft'
units_O_freq = 'GHz'
Tuned_freq = 2.45
Low_freq = 0.5
High_freq = 8

# Need to put in a case statedment for the unit_O_freq, right now treated it as
# GHz
def Hz():
    return 1
def kHz():
    return 1e3
def MHz():
    return 1e6
def GHz():
    return 1e9
    
# All calculations completed in Hz
freq_unit = {'Hz' : Hz,
           'kHz' : kHz,
           'MHz' : MHz,
           'GHz' : GHz
           }

def meter():
    return 1
def foot():
    return 100/2.54/12
def inch():
    return 100/2.54
def centimeter():
    return 100
def kilometer():
    return 0.001
def millimeter():
    return 1000

# all calculations will be completed in meters
length_unit = {'m' : meter,
               'ft' : foot,
               'in' : inch,
               'cm' : centimeter,
               'km' : kilometer,
               'mm' : millimeter
               }

Len_unit = length_unit[units_O_len]()   # get the unit conversion factor
