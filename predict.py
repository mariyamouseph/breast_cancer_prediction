from sklearn.svm import SVC

from helper import read_data, pre_process_data, visualize_data, visualize_correlation, split_data, check_accuracy, \
    plot_performance, standardize_model, tune_svm, prepare_model, calculate_accuracy

# Data preparation
raw_data = read_data()
processed_data = pre_process_data(raw_data)
visualize_data(processed_data)
visualize_correlation(processed_data)
x_train, x_test, y_train, y_test = split_data(processed_data)

# Checking initial accuracy
cv_results, name = check_accuracy(x_train, y_train, SVC(), 'SVM')
plot_performance(cv_results, name)


# Model tuning and checking accuracy
model = standardize_model()
std_cv_results, std_name = check_accuracy(x_train, y_train, model, 'SVM')
plot_performance(std_cv_results, std_name)
tune_svm(x_train, y_train)
model, scaler = prepare_model(x_train, y_train)
calculate_accuracy(scaler, model, x_test, y_test)

