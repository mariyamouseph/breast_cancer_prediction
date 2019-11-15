import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def read_data():
    """
    Read data from csv
    """

    data = pd.read_csv('input/data.csv', index_col=False)
    return data


def pre_process_data(data):
    """
    Setting 1 for malignant data and 0 for benign data
    """

    data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    data = data.set_index('id')
    del data['Unnamed: 32']
    return data


def visualize_data(data):
    """
    Printing the number of malignant and benign data and visualization using density plot
    """

    print(data.groupby('diagnosis').size())
    data.plot(kind='density', subplots=True, layout=(5, 7), sharex=False, legend=False, fontsize=1)
    plt.show()


def visualize_correlation(data):
    """
    Plotting attribute correlation
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(data.corr(), interpolation="none", cmap=cmap)
    ax1.grid(True)
    plt.title('Breast Cancer Attributes Correlation')
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()


def split_data(data):
    """
    Splitting to train and test data
    """

    y = data['diagnosis'].values
    x = data.drop('diagnosis', axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=21)
    return x_train, x_test, y_train, y_test


def check_accuracy(x_train, y_train, model, name):
    """
    Checking accuracy with passed model
    """
    num_folds = 10
    kfold = KFold(n_splits=num_folds, random_state=123)
    start = time.time()
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    print("%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
    return cv_results, name


def plot_performance(cv_results, name):
    """
    Plotting performance of algorithm
    """

    fig = plt.figure()
    fig.suptitle('Performance')
    ax = fig.add_subplot(111)
    plt.boxplot([cv_results])
    ax.set_xticklabels([name])
    plt.show()


def standardize_model():
    model = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])
    return model


def tune_svm(x_train, y_train):
    num_folds = 10
    scaler = StandardScaler().fit(x_train)
    rescaledx = scaler.transform(x_train)
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)
    model = SVC()
    kfold = KFold(n_splits=num_folds, random_state=21)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(rescaledx, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def prepare_model(x_train, y_train):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    model = SVC(C=2.0, kernel='rbf')
    start = time.time()
    model.fit(x_train_scaled, y_train)
    end = time.time()
    print("Run Time: %f" % (end - start))
    return model, scaler


def calculate_accuracy(scaler, model, x_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_test_scaled = scaler.transform(x_test)
    predictions = model.predict(x_test_scaled)
    print("Accuracy score %f" % accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
