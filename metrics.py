from utils import average
from sklearn import metrics
import wandb


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
        print(f"Accuracy: {average(self.accuracy, 4)}")
        print(f"Precision: {average(self.precision, 4)}")
        print(f"Recall / Sensitivity: {average(self.recall, 4)}")
        print(f"Specificity: {average(self.specificity, 4)}")
        print(f"Dice coefficient / F-score: {average(self.Dice, 4)}")
        print(f"Jaccard index / IoU: {average(self.Jaccard, 4)}")

        wandb.log({"Accuracy": average(self.accuracy, 4),
                  "Precision": average(self.precision, 4),
                   "Recall": average(self.recall, 4),
                   "Specificity": average(self.specificity, 4),
                   "Dice coefficient": average(self.Dice, 4),
                   "Jaccard index": average(self.Jaccard, 4)})
