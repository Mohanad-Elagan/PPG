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