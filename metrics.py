from numpy import unique
from utils import average
from sklearn import metrics


class SegmentationMetrics():
    def __init__(self):
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.specificity = []
        self.Dice = []
        self.Jaccard = []

    def evaluate_pair(self, y_true, y_pred):
        # Some sklearn metrics (like recall) require 1D arguments
        y_true = (y_true).flatten()
        y_pred = (y_pred / 255).flatten()

        self.accuracy.append(metrics.accuracy_score(y_true, y_pred))
        self.precision.append(metrics.precision_score(y_true, y_pred))
        self.recall.append(metrics.recall_score(y_true, y_pred))
        self.specificity.append(
            metrics.recall_score(y_true, y_pred, pos_label=0))
        self.Dice.append(metrics.f1_score(y_true, y_pred))
        self.Jaccard.append(metrics.jaccard_score(y_true, y_pred))

    def summary(self):
        print(f"Segmentation metrics")
        print(f"Accuracy: {round(self.accuracy[0], 4)}")
        print(f"Precision: {round(self.precision[0], 4)}")
        print(f"Recall / Sensitivity: {round(self.recall[0], 4)}")
        print(f"Specificity: {round(self.specificity[0], 4)}")
        print(f"Dice coefficient / F-score: {round(self.Dice[0], 4)}")
        print(f"Jaccard index / IoU: {round(self.Jaccard[0], 4)}")
