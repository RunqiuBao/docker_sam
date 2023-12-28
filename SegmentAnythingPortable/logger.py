from utils import CreatePrintableTable
import os
import torch
from datetime import datetime

class Logger(object):
    _trainLosses = None
    _validLosses = None
    _indexBestEpoch = None
    _logId = None
    _logPath = None

    def __init__(self, logPath='./logs/'):
        self._trainLosses, self._validLosses = [], []
        self._indexBestEpoch = 0
        now = datetime.now()
        self._logId = now.strftime("log-%Y_%m_%d_%H_%M_%S")
        if not os.path.isdir(logPath):
            os.makedirs(logPath)
        self._logPath = logPath

    def Record(self, metrics, state, phase):
        if phase == 'train':
            self._trainLosses.append(metrics)
        elif phase == 'valid':
            self._validLosses.append(metrics)
            if metrics < self._validLosses[self._indexBestEpoch]:
                self._indexBestEpoch = len(self._validLosses) - 1
                self.SaveState(state, filename='best.pth.tar')

    def PrintReport(self):
        sReportHeaders = [["logId"], ["number of epoches"], ["index best epoch"], ["best metrics"]]
        sReportRows = [[self._logId], [len(self._trainLosses)], [self._indexBestEpoch], [self._validLosses[self._indexBestEpoch]]]
        print(CreatePrintableTable(sReportRows, sReportHeaders))

    def SaveState(self, state, filename='checkpoint.pth.tar'):
        logSavePath = os.path.join(self._logPath, filename)
        torch.save(state, logSavePath)

