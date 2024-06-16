import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Load the data
df = pd.read_csv('rawdata.csv')
# Separate the samples and the wavelength data
samples = df.iloc[:, 0]
wavelengths = df.columns[1:].astype(float)
data = df.iloc[:, 1:].values
# Baseline correction function using Asymmetric Least Squares Smoothing
def baseline_als(y, lam=1e6, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z  # Return the corrected signal
  # Apply baseline correction to each sample
corrected_data = np.apply_along_axis(baseline_als, 1, data)
# Plot the corrected spectra
plt.figure(figsize=(10, 6))
for i in range(corrected_data.shape[0]):
    plt.plot(wavelengths, corrected_data[i, :], label=samples[i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Corrected Spectra')
plt.legend()
plt.show()
# Example function to perform polynomial baseline correction
def polynomial_baseline_correction(data, degree=3):
    x = np.arange(data.shape[1])
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    lin_reg = LinearRegression()
    baselines = []
    for spectrum in data:
        lin_reg.fit(X_poly, spectrum)
        baseline = lin_reg.predict(X_poly)
        baselines.append(baseline)
    return np.array(baselines)
  # Apply polynomial baseline correction
polynomial_baseline = polynomial_baseline_correction(data)
corrected_data_polynomial = data - polynomial_baseline
from scipy.signal import savgol_filter

# Example function to perform Savitzky-Golay filter baseline correction
def savgol_baseline_correction(data, window_length=11, polyorder=3):
    baselines = savgol_filter(data, window_length, polyorder, axis=1)
    return baselines

# Apply Savitzky-Golay filter baseline correction
savgol_baseline = savgol_baseline_correction(data)
corrected_data_savgol = data - savgol_baseline
from scipy.linalg import cho_factor, cho_solve

# Example function to perform Whittaker smoothing baseline correction
def whittaker_smooth(x, w, lambda_, differences=1):
    X = np.array(x)
    m = X.shape[0]
    E = np.eye(m)
    D = np.diff(E, differences).T
    W = np.diag(w)
    A = W + lambda_ * np.dot(D.T, D)
    b = np.dot(W, X)
    cho = cho_factor(A)
    z = cho_solve(cho, b)
    return z

def whittaker_baseline_correction(data, lambda_=1e5, p=0.1):
    baselines = []
    for spectrum in data:
        w = np.ones(len(spectrum))
        w[spectrum < np.percentile(spectrum, p*100)] = p
        baseline = whittaker_smooth(spectrum, w, lambda_)
        baselines.append(baseline)
    return np.array(baselines)

# Apply Whittaker smoothing baseline correction
whittaker_baseline = whittaker_baseline_correction(data)
corrected_data_whittaker = data - whittaker_baseline
from scipy.interpolate import UnivariateSpline

# Example function to perform spline fitting baseline correction
def spline_baseline_correction(data, s=1e-5):
    x = np.arange(data.shape[1])
    baselines = []
    for spectrum in data:
        spline = UnivariateSpline(x, spectrum, s=s)
        baseline = spline(x)
        baselines.append(baseline)
    return np.array(baselines)

# Apply spline fitting baseline correction
spline_baseline = spline_baseline_correction(data)
corrected_data_spline = data - spline_baseline
# Plotting function
def plot_spectra(wavelengths, original, baseline, corrected, title):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, original, label='Original Spectrum')
    plt.plot(wavelengths, baseline, label='Baseline', linestyle='--')
    plt.plot(wavelengths, corrected, label='Corrected Spectrum')
    plt.title(title)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
  # Select a sample index to visualize
sample_index = 0
# Plot polynomial correction
plot_spectra(wavelengths, data[sample_index], polynomial_baseline[sample_index], corrected_data_polynomial[sample_index], 'Polynomial Baseline Correction')
# Plot Savitzky-Golay correction
plot_spectra(wavelengths, data[sample_index], savgol_baseline[sample_index], corrected_data_savgol[sample_index], 'Savitzky-Golay Baseline Correction')
# Plot Whittaker smoothing correction
plot_spectra(wavelengths, data[sample_index], whittaker_baseline[sample_index], corrected_data_whittaker[sample_index], 'Whittaker Smoothing Baseline Correction')
# Plot spline fitting correction
plot_spectra(wavelengths, data[sample_index], spline_baseline[sample_index], corrected_data_spline[sample_index], 'Spline Fitting Baseline Correction')
