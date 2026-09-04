import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

date = datetime(2024, 5, 31, 16, 0, tzinfo=EASTERN)

from features_and_labels import get_labels, get_features

STOCKS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'FB', 'NVDA', 'NFLX', 'INTC', 'AMD']
STARTDATE = datetime(2024, 6, 1)
TRAINING_ENDDATE = datetime(2026, 8, 31)
X = []
Y = []

for days in range((TRAINING_ENDDATE - STARTDATE).days):
    for stock in STOCKS:
        date = STARTDATE + timedelta(days=days)
        labels = get_labels(stock, date)
        features = get_features(stock, date)
        X.append(features)
        input(features.shape)
        Y.append(labels)

