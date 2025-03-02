import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from tqdm import tqdm
import sys
import scipy as sp
from numba import njit

def do_nothing():
    pass

def random_color():
    return [int(np.random.uniform(100, 255,1)[0]), int(np.random.uniform(100, 255,1)[0]), int(np.random.uniform(100, 255,1)[0])]

@njit
def fact(x):
    out = 1
    for i in range(x):
        out *= (x-i)
    return out

def sigmoid(x, x0, k, A):
     y = A / (1 + np.exp(-k*(x-x0)))
     return y

def convolve_with_exp(ts, tau):
    t = np.linspace(0, ts[-1], ts.shape[0])
    return sp.signal.convolve(ts, exponential(t, tau))[:len(t)]

@njit
def exponential(t, tau):
    return np.exp(-1 * (t / tau))

@njit
def compute_phase_coherence_matrix(ts):
    nodes_n_rem = ts.shape[0]
    C_ij = np.zeros((nodes_n_rem, nodes_n_rem))
    for i in range(nodes_n_rem):
        for j in range(i + 1, nodes_n_rem):
            C_ij[i,j] = np.abs(np.mean(np.exp(1j * (ts[i,:] - ts[j,:]))))
    return C_ij+C_ij.T

@njit
def compute_phase_coherence(signal1, signal2):
    complex_phase_difference = np.exp(1j * (signal1 - signal2))
    R = np.abs(np.mean(complex_phase_difference))
    return R

def compute_order(ts, series_out=True): # on prend en argument une liste de séries temporelles
    order_series = np.mean(np.exp(1j*ts), axis=0)
    if series_out:
        return np.abs(order_series)
    else:
        return np.abs(np.mean(order_series))

def identify_files(path, keywords=None, exclude=None):
    items = os.listdir(path)
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            if any(excluded in item for excluded in exclude):
                pass
            else:
                files.append(item)
    files.sort()
    return files

def load_hdf5(path):
    data = {}
    file = h5py.File(path, 'r')
    for dataset in file.keys():
        data[dataset] = np.array(file[dataset])
    file.close()
    return data

def save_hdf5(path, dictionary):
    datasets = list(dictionary.keys())
    file = h5py.File(path, 'w')
    for dataset in datasets:
        file.create_dataset(dataset, data=dictionary[dataset])
    file.close()

def baseline_minfilter(signal, window=300, sigma1=5, sigma2=100, debug=False):
    signal_flatstart = np.copy(signal)
    signal_flatstart[0] = signal[1]
    smooth = sp.ndimage.gaussian_filter1d(signal_flatstart, sigma1)
    mins = sp.ndimage.minimum_filter1d(smooth, window)
    baseline = sp.ndimage.gaussian_filter1d(mins, sigma2)
    if debug:
        debug_out = np.asarray([smooth, mins, baseline])
        return debug_out
    else:
        return baseline


def compute_dff_using_minfilter(timeseries, window=200, sigma1=0.1, sigma2=50):
    dff = np.zeros(timeseries.shape)
    for i in range(timeseries.shape[0]):
        if np.any(timeseries[i]):
            baseline = baseline_minfilter(timeseries[i], window=window, sigma1=sigma1, sigma2=sigma2)
            dff[i] = (timeseries[i] - baseline) / (baseline+1)
    return dff


@njit
def compute_correlation_matrix(timeseries, arctanh=False):
    N = timeseries.shape[0]
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if arctanh:
                matrix[i, j] = np.arctanh(np.corrcoef(timeseries[i], timeseries[j])[0, 1])
            else:
                matrix[i, j] = np.corrcoef(timeseries[i], timeseries[j])[0, 1]
            matrix[j, i] = matrix[i, j]
    return matrix


def correlate_matrices(matrix1, matrix2, choice=True):
    triangle = np.triu_indices(matrix1.shape[0], 1)
    r1 = sp.stats.pearsonr(matrix1[triangle], matrix2[triangle])[0]
    r2 = sp.stats.spearmanr(matrix1[triangle], matrix2[triangle])[0]
    r = [r1, r2]
    if choice:
        return r[np.argmax(r)]
    else:
        return r1

def delete(array, deleted):
    truncated = np.copy(array)
    truncated = np.delete(truncated, deleted, axis=0)
    truncated = np.delete(truncated, deleted, axis=1)
    return truncated

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

def fill_index(id):
    if len(str(id)) == 1:
        return "0" + str(id)
    else:
        return str(id)

def colors(n):
    couleurs = ["aqua", 
                "aquamarine", 
                "azure", 
                "beige", 
                "black", 
                "blue",
                "brown",
                "chartreuse",
                "chocolate",
                "coral",
                "crimson",
                "cyan", 
                "darkblue",
                "darkgreen", 
                "fuchsia",
                "gold",
                "goldenrod",
                "green",
                "grey",
                "indigo",
                "ivory",
                "khaki",
                "lavender",
                "lightblue",
                "lightgreen",
                "lime",
                "magenta", 
                "maroon",
                "navy",
                "olive",
                "orange",
                "orangered",
                "orchid",
                "pink",
                "plum",
                "purple",
                "red",
                "salmon",
                "sienna",
                "silver",
                "tan",
                "teal",
                "tomato",
                "turquoise",
                "violet",
                "wheat",
                "white",
                "yellow",
                "yellowgreen"
               ]

    return couleurs[0:n]
