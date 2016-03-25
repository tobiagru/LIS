# Images
Learning and Intelligent Systems - Project 3: <http://las.ethz.ch/courses/lis-s15/>

## Installing libraries
We have to install Lasagne from source as there are no PiPy packages yet.
If you do not use Anaconda, then just use `pip` instead of `$HOME/anaconda/bin/pip`.

    # clone this into some folder
    git clone git@github.com:Lasagne/Lasagne.git
    cd Lasagne
    $HOME/anaconda/bin/pip install nolearn
    $HOME/anaconda/bin/pip install -r requirements.txt
    $HOME/anaconda/bin/python setup.py install
    cd ..; rm -rf Lasagne

    git clone https://github.com/cudamat/cudamat.git
    cd cudamat
    PATH="${PATH}:/Developer/NVIDIA/CUDA-7.0/bin" $HOME/anaconda/bin/python setup.py install
    cd ..; rm -rf cudamat

    # should print 'theano'
    $HOME/anaconda/bin/python -c 'import theano; print theano.__name__'

Now, we have to enable GPU support.

* [Linux](http://deeplearning.net/software/theano/install.html#gpu-linux)
* [OS X](http://deeplearning.net/software/theano/install.html#gpu-macos)
* [Windows](http://deeplearning.net/software/theano/install_windows.html#gpu-windows)

After installing, try to run `test_gpu.py`:

    ./run.sh test_gpu.py

If "Used the gpu" is printed at the end, everything works now.

**Warning:** the training and result data must not be checked into this repo (ETH Copyright).
