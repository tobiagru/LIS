# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:42:25 2015

@author: Andreas
"""

import pandas as pd
import numpy as np
from lib import *


Yreal = load_Y('train')
Ypred = load_Y('test_reality')

real=Yreal['y2']
pred=Ypred['y2']


#Diagram: What should it be, what was recogniced
nr_of_labels = real.unique().size
cout_labels = ((real)*(nr_of_labels) + pred)/nr_of_labels +1 #eg 0...one was is true and was predicted




print cout_labels


cout_labels.hist(bins=nr_of_labels**2, )
#Label: Diagram ... shows for each label which label was predicted the most.

