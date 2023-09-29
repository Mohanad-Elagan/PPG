from models.pytorch_nn_regression import run_regression
from splitted_sbp_dbp_features import getFeatures_SBP_DBP, normalize, add_intercept, get_labels
from sklearn.model_selection import train_test_split
from feature_select import get_filtered_features_reg, get_filtered_features_pearsons, get_filtered_features_random_forest_reg
import torch.nn as nn
import utils
import numpy as np
import os
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
num_features = [28, 16, 10, 6, 3]
data = getFeatures_SBP_DBP(all_ppg=True)
features = data["features"]
sbp = data["sbp"]
dbp = data["dbp"]
features = normalize(features)

sbp_train_accuracies = []
dbp_train_accuracies = []

sbp_test_accuracies = []
dbp_test_accuracies = []

sbp_train_accuracies_sd = []
dbp_train_accuracies_sd = []

sbp_test_accuracies_sd = []
dbp_test_accuracies_sd = []

classification_accuracies_multiclass = []
classification_accuracies_binary = []

for num_feature in num_features:
    sbp_features = get_filtered_features_pearsons(features, sbp, num_feature, "sbp")
    dbp_features = get_filtered_features_pearsons(features, dbp, num_feature, "dbp")

    train_x_sbp, test_x_sbp, train_targets_sbp, test_targets_sbp, train_x_dbp, test_x_dbp, train_targets_dbp, test_targets_dbp = train_test_split(sbp_features, sbp, dbp_features ,dbp, train_size=0.7)

    e_train_accuracy_epochs, e_test_accuracy_epochs, sbp_predicted_test, ae_sbp_train, ae_sbp_test, sbp_error_train, sbp_error_test = run_regression(train_x_sbp, train_targets_sbp, test_x_sbp, test_targets_sbp,learning_rate=0.005,num_epochs=1000, verbose=False)
    e_train_accuracy_epochs, e_test_accuracy_epochs, dbp_predicted_test, ae_dbp_train, ae_dbp_test, dbp_error_train, dbp_error_test = run_regression(train_x_dbp, train_targets_dbp, test_x_dbp, test_targets_dbp,learning_rate=0.005,num_epochs=1000, verbose=False)

    sbp_train_accuracies.append(np.mean(ae_sbp_train))
    dbp_train_accuracies.append(np.mean(ae_dbp_train))
    sbp_train_accuracies_sd.append(np.std(ae_sbp_train))
    dbp_train_accuracies_sd.append(np.std(ae_dbp_train))
    sbp_test_accuracies.append(np.mean(ae_sbp_test))
    dbp_test_accuracies.append(np.mean(ae_dbp_test))
    sbp_test_accuracies_sd.append(np.std(ae_sbp_test))
    dbp_test_accuracies_sd.append(np.std(ae_dbp_test))

    multiclass_predicted, binary_predicted = get_labels(sbp_predicted_test, dbp_predicted_test)
    multiclass_true, binary_true = get_labels(test_targets_sbp, test_targets_dbp)

    classification_accuracies_multiclass.append(1 - (np.count_nonzero(np.abs(multiclass_predicted-multiclass_true))/multiclass_true.shape[0]))
    classification_accuracies_binary.append(1 - (np.count_nonzero(np.abs(binary_predicted-binary_true))/binary_true.shape[0]))

    figure1, axis = plt.subplots(1,2)
    axis[0].hist(sbp_error_test, bins=30, rwidth=0.75)
    axis[1].hist(dbp_error_test, bins=30, rwidth=0.75)
    axis[0].set_xlabel("True value - Predicted value")
    axis[1].set_xlabel("True value - Predicted value")
    axis[0].set_ylabel("# test examples")
    axis[1].set_ylabel("# test examples")
    figure1.tight_layout()
    figure1.savefig(f"nn_error_distribution_num_features{num_feature}.png")

figure2, axis2 = plt.subplots(1,2)
axis2[0].plot(np.array(num_features), np.array(sbp_train_accuracies), label="Train MAE")
axis2[0].plot(np.array(num_features), np.array(sbp_test_accuracies), label="Test MAE")

axis2[1].plot(np.array(num_features), np.array(dbp_train_accuracies))
axis2[1].plot(np.array(num_features), np.array(dbp_test_accuracies))

axis2[0].set_xlabel("# features")
axis2[0].set_ylabel("MAE")

axis2[1].set_xlabel("# features")
axis2[1].set_ylabel("MAE")

axis2[0].set_xticks(np.array(num_features))
axis2[1].set_xticks(np.array(num_features))

figure2.tight_layout()
figure2.legend()
figure2.savefig(f"nn_reg_accuracies.png")

np.savetxt("nnreg_sbp_train_MAE.csv", np.array(sbp_train_accuracies), delimiter=",")
np.savetxt("nnreg_sbp_test_MAE.csv", np.array(sbp_test_accuracies), delimiter=",")
np.savetxt("nnreg_sbp_train_MAE_sd.csv", np.array(sbp_train_accuracies_sd), delimiter=",")
np.savetxt("nnreg_sbp_test_MAE_sd.csv", np.array(sbp_test_accuracies_sd), delimiter=",")
np.savetxt("nnreg_dbp_train_MAE.csv", np.array(dbp_train_accuracies), delimiter=",")
np.savetxt("nnreg_dbp_test_MAE.csv", np.array(dbp_test_accuracies), delimiter=",")
np.savetxt("nnreg_dbp_train_MAE_sd.csv", np.array(dbp_train_accuracies_sd), delimiter=",")
np.savetxt("nnreg_dbp_test_MAE_sd.csv", np.array(dbp_test_accuracies_sd), delimiter=",")
np.savetxt("nnreg_multiclass_classification_accuracy.csv", np.array(classification_accuracies_multiclass), delimiter=",")
np.savetxt("nnreg_binary_classification_accuracy.csv", np.array(classification_accuracies_binary), delimiter=",")