import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, ExpSineSquared, RationalQuadratic
)
from scipy.stats import interp1d

