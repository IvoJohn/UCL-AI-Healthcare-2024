from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import numpy as np
import cv2
import tensorflow
import os


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
        axs[0].set_ylim([0, np.max(np.concatenate(([1], self.losses, self.val_losses), axis=0))])
        axs[1].set_ylim([0, 1])
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
    
    print("\nProvide patient's parameters\n")
    
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

    inputs = [hr, o2, sbp, map, resp, age,
                   gender, hosp_adm, iculos]
    inputs = [float(x) for x in inputs]

    x = np.reshape(inputs, (1, 9))
    pred = model.predict(x, verbose=0)
    if pred > threshold:
        print('\n\nPREDICTION: Sepsis')
    else:
        print('\n\nPREDICTION: Healthy')


def patient_predict_tree(tree_model, threshold=0.5):
    
    print("\nProvide patient's parameters\n")
    
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

    inputs = [hr, o2, sbp, map, resp, age,
                    gender, hosp_adm, iculos]
    inputs = [float(x) for x in inputs]

    x = np.reshape(inputs, (1, 9))
    pred = tree_model.predict(x)
    if pred > threshold:
        print('\n\nPREDICTION: Sepsis')
    else:
        print('\n\nPREDICTION: Healthy')


def evaluate_model_first(model, test_directory, img_size=(192,192), batch_size=24):
    test_data = tensorflow.keras.utils.image_dataset_from_directory(
    directory=test_directory,
    seed=1,
    image_size=img_size,
    batch_size=batch_size
    )

    model_evaluation = model.evaluate(test_data)

    print('\nTEST RESULTS')
    print('Accuracy: {:.4f}'.format(model_evaluation[1]))
    print('Area under a receiver operating characteristic curve: {:.4f}'.format(model_evaluation[2]))
    print('False negatives: {:.0f}/{:.0f}'.format(model_evaluation[3],len(test_data)*batch_size))
    print('False positives: {:.0f}/{:.0f}'.format(model_evaluation[4],len(test_data)*batch_size))
    print('True negatives: {:.0f}/{:.0f}'.format(model_evaluation[5],len(test_data)*batch_size))
    print('True positives: {:.0f}/{:.0f}'.format(model_evaluation[6],len(test_data)*batch_size))
    print('Recall: {:.4f}'.format(model_evaluation[7]))

        
def show_predictions_first(test_path):
    false_positive_path = os.path.join(test_path, 'benign/1537.jpg') #undetected benign
    true_negative_path = os.path.join(test_path, 'benign/1025.jpg') #detected benign
    false_negative_path = os.path.join(test_path, 'malignant/1380.jpg') # undetected malignant
    true_positive_path = os.path.join(test_path, 'malignant/953.jpg') #detected malignant

    fig, axs = plt.subplots(2,2, figsize=(6,6), dpi=100)

    axs[0,0].imshow(plt.imread(true_positive_path))
    axs[0,0].set_title('True positive', size=15)
    axs[0,0].axis('off')

    axs[0,1].imshow(plt.imread(false_negative_path))
    axs[0,1].set_title('False negative', size=15)
    axs[0,1].axis('off')

    axs[1,0].imshow(plt.imread(false_positive_path))
    axs[1,0].set_title('False positive', size=15)
    axs[1,0].axis('off')

    axs[1,1].imshow(plt.imread(true_negative_path))
    axs[1,1].set_title('True negative', size=15)
    axs[1,1].axis('off')

    plt.show()

    
def show_other():

    #n source - https://www.clevelandclinicmeded.com/medicalpubs/diseasemanagement/dermatology/common-benign-growths/
    #p source - 

    different_img_n = '/content/00.png'
    different_img_p = '/content/11.png'

    im_n = cv2.cvtColor(cv2.imread(different_img_n), cv2.COLOR_BGR2RGB)
    im_p = cv2.cvtColor(cv2.imread(different_img_p), cv2.COLOR_BGR2RGB)

    pred_n = np.expand_dims(im_n, axis=0)
    pred_p = np.expand_dims(im_p, axis=0)

    fig, axs = plt.subplots(1,2, figsize=(6,3))

    axs[0].imshow(im_n)
    axs[0].axis('off')
    axs[0].set_title('Benign \nclassified as malignant', size=15)

    axs[1].imshow(im_p)
    axs[1].axis('off')
    axs[1].set_title('Malignant \nclassified as benign', size=15)
