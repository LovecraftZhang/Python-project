import numpy as np
import scipy 
import pickle
import sys 

def mat2pkl(pkl_path, mat_path):
    x = scipy.io.loadmat(mat_path, squeeze_me=True)
    del x['__header__']
    del x['__version__']
    del x['__globals__']

    with open(pkl_path, 'wb') as f:
        pickle.dump(x, f, protocol=2)

    f.close()

    print("%s saved..." % pkl_path)

def save_pkl(fname, data):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    fname = fname.replace(".pickle", "")
    fname += ".pickle"

    with open(fname, 'wb') as fl:
        pickle.dump(data, fl, protocol=2)

    print "%s saved..." % fname

    return data

def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data


def to_categorical(y):
    n_values = np.max(y) + 1
    
    return np.eye(n_values)[y]