#H. Wang, “Random forest based blood pressure prediction model from ecg and ppg signal,” in 2022 12th International Conference on Bioscience, Biochemistry and Bioinformatics, 2022, pp. 1–6.
#S. Janitza, G. Tutz, and A.-L. Boulesteix, “Random forest for ordinal responses: prediction and variable selection,” Computational Statistics & Data Analysis, vol. 96, pp. 57–73, 2016.
#H. Han, X. Guo, and H. Yu, “Variable selection using mean decrease accuracy and mean decrease gini based on random forest,” in 2016 7th ieee international conference on software engineering and service science (icsess). IEEE, 2016, pp. 219–224.
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from splitted_sbp_dbp_features import *
from feature_select import *
import pydot

data = getFeatures_SBP_DBP(all_ppg=True)
features = data["features"]
sbp = data["sbp"]
dbp = data["dbp"]
features = get_filtered_features_random_forest_reg(features, sbp)

X, y = features, sbp

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=1, random_state=42)
rf.fit(X_train, y_train)

tree = rf.estimators_[0]

export_graphviz(tree, out_file="tree.dot",
                feature_names=range(1, X.shape[1] + 1),
                rounded=True,
                filled=True)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

Image(filename='tree.png')