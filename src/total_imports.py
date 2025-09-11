from numpy import zeros, reshape, putmask
import pickle
import os
import json
from ReadObs import ReadObs
from ReadParams import ReadParams
from ReadTruth import ReadTruth
from SelObs import SelObs
from CalcdA import CalcdA
from ProcessPrior import ProcessPrior
from GetCovMats import GetCovMats
from MetropolisCalculations import MetropolisCalculations
from CalculateEstimates import CalculateEstimates
from MakeFigs import MakeFigs
from CalcErrorStats import CalcErrorStats
from FilterEstimate import FilterEstimate
from DispRMSEStats import DispRMSEStats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Getsos_calibration import Getsos_calibration
from ReadS3 import ReadS3
from CalculateEstimates_hydrocorn import CalculateEstimates_hydrocorn
from Recalculate_product import Recalculate_product
from find_peak_position import find_peak_position
from scipy.stats import pearsonr,spearmanr,truncnorm, norm,gaussian_kde
from scipy.signal import find_peaks