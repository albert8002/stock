import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

STOCKS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'FB', 'NVDA', 'NFLX', 'INTC', 'AMD']
TIMESPANS = [1, 3, 7, 15, 30]  # in days

feat