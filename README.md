# UV-Vis-NIR Spectroscopy Analysis

This repository contains data and analysis scripts for UV-Vis-NIR spectroscopy using the Simadzhu 1800 model. The analysis involves baseline correction and detrending with a 2nd order polynomial transformation.

## Project Structure

- `data/`: Contains the processed CSV files.
- `scripts/`: Contains the Python scripts used for analysis.
- `results/`: Plots and results from the analysis.

## Data

The data is split into three segments:
- `UVsegment`: 190-400 nm
- `VISsegment`: 401-800 nm
- `NIRsegment`: 801-1100 nm

Each segment has been processed using two methods:
- Adoptive baseline correction
- Detrending with a 2nd order polynomial

## Analysis

The analysis includes descriptive statistics and Principal Component Analysis (PCA) to visualize the data. The steps are as follows:

1. Load the data.
2. Perform descriptive statistics.
3. Conduct PCA and plot the results.

## Usage

To run the analysis, execute the scripts in the `scripts/` directory.

## Dependencies

- pandas
- seaborn
- matplotlib
- scipy
- scikit-learn

=====
