from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import numpy as np


class PlotLossesFirst(keras.callbacks.Callback):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.recalls = []
        self.val_recalls = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        val_recall = [x for x in list(
            logs.keys()) if x.startswith('recall')][0]
        val_recall_key = [x for x in list(
            logs.keys()) if x.startswith('val_recall')][0]
        self.recalls.append(logs.get(val_recall))
        self.val_recalls.append(logs.get(val_recall_key))

        #print('Logs keys',logs.keys())

        # recall = logs.get('true_positives') / (logs.get('true_positives')+logs.get('false_positives')

        self.i += 1

        clear_output(wait=True)

        # add here model prediction and title with probability
        # of it - one for malignant and one for bening from validation dataset
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=100)
        axs[0].set_xlim([0, self.n_epochs-1])
        axs[1].set_xlim([0, self.n_epochs-1])
        axs[0].plot(self.x, self.losses, label="Training loss")
        axs[0].plot(self.x, self.val_losses, label="Validaiton loss")
        axs[1].plot(self.x, self.recalls, label="Training recall")
        axs[1].plot(self.x, self.val_recalls, label="Validation recall")
        axs[0].set_title('Loss')
        axs[1].set_title('Recall')
        # axs.set_title(logs.keys())
        axs[0].legend()
        axs[1].legend()
        plt.show()


class PlotLossesSecond(keras.callbacks.Callback):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.accuracy = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

        self.i += 1

        clear_output(wait=True)
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=100)
        axs[0].set_xlim([0, self.n_epochs-1])
        axs[1].set_xlim([0, self.n_epochs-1])
        axs[0].set_ylim([0, 1])
        axs[1].set_ylim([0, 1])
        axs[0].plot(self.x, self.losses, label="Training loss")
        axs[1].plot(self.x, self.accuracy, label="Training accuracy")
        axs[0].set_title('Loss')
        axs[1].set_title('Accuracy')
        axs[0].legend()
        axs[1].legend()
        plt.show()


def test_show_metrics_dnn(model, X_test, y_test, threshold=0.5, labels=['Healthy', 'Sepsis']):
    y_pred = model.predict(X_test)

    threshold = 0.5
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    conf = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                   display_labels=labels,
                                                   cmap=plt.cm.Blues)
    fig = conf.figure_
    fig.set_figwidth(7)
    fig.set_figheight(7)
    fig.suptitle('Plot of confusion matrix for DNN')
    fig.tight_layout()
    plt.show()

    print('Accuracy: {:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('Recall: {:.3f}'.format(metrics.recall_score(y_test, y_pred)))
    print('Precission: {:.3f}'.format(metrics.precision_score(y_test, y_pred)))


def test_show_metrics_tree(tree_model, X_test, y_test, threshold=0.5, labels=['Healthy', 'Sepsis']):
    y_pred = tree_model.predict(X_test)

    threshold = 0.5
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    conf = ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                   display_labels=labels,
                                                   cmap=plt.cm.Blues)

    fig = conf.figure_
    fig.set_figwidth(7)
    fig.set_figheight(7)
    fig.suptitle('Plot of confusion matrix for DNN')
    fig.tight_layout()
    plt.show()

    print('Accuracy: {:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('Recall: {:.3f}'.format(metrics.recall_score(y_test, y_pred)))
    print('Precission: {:.3f}'.format(metrics.precision_score(y_test, y_pred)))


def patient_predict_dnn(model, threshold=0.5):
    hr = input('HR [30-200]: ')
    o2 = input('Saturation [40-100]: ')
    sbp = input('Systolic blood pressure [35-240]: ')
    map = input('Mean Arterial Pressure [20-240]: ')
    resp = input('Breaths per minute [1-60]: ')
    age = input('Age [20-100]: ')
    gender = input('Men [0], Woman [1]: ')
    hosp_adm = input(
        'Hours between hospital admit and ICU admit [-1000 - 0]: ')
    iculos = input('Length of stay in ICU in hours []: [5-150]')

    x = np.reshape([hr, o2, sbp, map, resp, age,
                   gender, hosp_adm, iculos], (1, 9))
    pred = model.predict(x, verbose=0)
    if pred > threshold:
        print('Sepsis')
    else:
        print('Healthy')


def patient_predict_tree(tree_model, threshold=0.5):
    hr = input('HR [30-200]: ')
    o2 = input('Saturation [40-100]: ')
    sbp = input('Systolic blood pressure [35-240]: ')
    map = input('Mean Arterial Pressure [20-240]: ')
    resp = input('Breaths per minute [1-60]: ')
    age = input('Age [20-100]: ')
    gender = input('Men [0], Woman [1]: ')
    hosp_adm = input(
        'Hours between hospital admit and ICU admit [-1000 - 0]: ')
    iculos = input('Length of stay in ICU in hours []: [5-150]')

    x = np.reshape([hr, o2, sbp, map, resp, age,
                   gender, hosp_adm, iculos], (1, 9))
    pred = tree_model.predict(x, verbose=0)
    if pred > threshold:
        print('Sepsis')
    else:
        print('Healthy')
