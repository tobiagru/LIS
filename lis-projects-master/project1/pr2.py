def load_X(fname):
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    names += ['K%02d' % i for i in range(1, 4 + 1)]
    names += ['L%02d' % i for i in range(1, 40 + 1)]
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       dtype=np.float64,
                       header=None,
                       names=names)
    return data

def load_Y(fname):
    return pd.read_csv('data/%s_y.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['y1', 'y2'])

def write_Y(fname, Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/%s_y_pred.txt' % fname, Y, fmt='%d', delimiter=',')