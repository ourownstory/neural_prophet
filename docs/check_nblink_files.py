# This script checks whether tutorial files have a corresponding nblink file and creates one if neccessary

import os

# get list of nblink files
nblink_list = os.listdir("./docs/source/example_links")
nblink_list_edit = []
for nblink_file in nblink_list:
    if nblink_file.endswith(".nblink"):
        nblink_list_edit.append(nblink_file[:-7])

# get list of tutorial files inside folder feature-use
feature_tutorial_list = os.listdir("./tutorials/feature-use")
feature_tutorial_list_edit = []
for feature_tutorial in feature_tutorial_list:
    if feature_tutorial.endswith(".ipynb"):
        feature_tutorial_list_edit.append(feature_tutorial[:-6])

# iterate through feature-use tutorial files and create nb-link file if necessary
for feature_tutorial in feature_tutorial_list_edit:
    if feature_tutorial not in nblink_list_edit:
        with open("./docs/source/example_links/" + feature_tutorial + ".nblink", "w") as out:
            line1 = "{"
            line2 = '    "path": "../../../tutorials/feature-use/' + feature_tutorial + '.nblink"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))

# get list of tutorial files inside folder application-example
app_tutorial_list = os.listdir("./tutorials/application-example")
app_tutorial_list_edit = []
for app_tutorial in app_tutorial_list:
    if app_tutorial.endswith(".ipynb"):
        app_tutorial_list_edit.append(app_tutorial[:-6])

# iterate through application-example tutorial files and create nb-link file if necessary
for app_tutorial in app_tutorial_list_edit:
    if app_tutorial not in nblink_list_edit:
        with open("./docs/source/example_links/" + app_tutorial + ".nblink", "w") as out:
            line1 = "{"
            line2 = '    "path": "../../../tutorials/application-example/' + app_tutorial + '.nblink"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))
