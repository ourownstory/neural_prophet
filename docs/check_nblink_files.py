# This script checks whether tutorial files have a corresponding nblink file and creates one if neccessary

import os
import pathlib

# define relative paths
DIR = pathlib.Path(__file__).parent.parent.absolute()
SRC_DIR = os.path.join(DIR, "docs", "source")
FEAT_TUT_DIR = os.path.join(DIR, "tutorials", "feature-use")
APP_TUT_DIR = os.path.join(DIR, "tutorials", "application-example")

# get list of nblink files and remove all .nblink files
nblink_list = os.listdir(SRC_DIR)
for nblink_file in nblink_list:
    if nblink_file.endswith(".nblink"):
        os.remove(os.path.join(SRC_DIR, nblink_file))
        pass

# get list of feature-use tutorial files and generate respective .nblink files
feature_tutorial_list = os.listdir(FEAT_TUT_DIR)
for feature_tutorial in feature_tutorial_list:
    if feature_tutorial.endswith(".ipynb"):
        # iterate through feature-use tutorial files and create nb-link file
        filename = feature_tutorial[:-6] + ".nblink"
        with open(os.path.join(SRC_DIR, filename), "w") as out:
            line1 = "{"
            line2 = '    "path": "../../tutorials/feature-use/' + feature_tutorial[:-6] + '.ipynb"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))

# get list of application-example tutorial files and generate respective .nblink files
app_tutorial_list = os.listdir(APP_TUT_DIR)
for app_tutorial in app_tutorial_list:
    if app_tutorial.endswith(".ipynb"):
        # iterate through application-example tutorial files and create nb-link file
        filename = app_tutorial[:-6] + ".nblink"
        with open(os.path.join(SRC_DIR, filename), "w") as out:
            line1 = "{"
            line2 = '    "path": "../../tutorials/application-example/' + app_tutorial[:-6] + '.ipynb"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))
