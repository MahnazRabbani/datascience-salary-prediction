import pandas as pd
import torch as t
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv('data/Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    return x, y

def get