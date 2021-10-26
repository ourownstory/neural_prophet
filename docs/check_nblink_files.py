# This script checks whether tutorial files have a corresponding nblink file and creates one if neccessary

import os

# get list of nblink files
nblink_list = os.listdir("./docs/source/example_links")
for nblink_file in nblink_list:
    if nblink_file.endswith(".nblink"):
        # TODO: Delete
        pass

# get list of tutorial files inside folder feature-use
feature_tutorial_list = os.listdir("./tutorials/feature-use")
for feature_tutorial in feature_tutorial_list:
    if feature_tutorial.endswith(".ipynb"):
        # iterate through feature-use tutorial files and create nb-link file
        with open("./docs/source/example_links/" + feature_tutorial[:-6] + ".nblink", "w") as out:
            line1 = "{"
            line2 = '    "path": "../../../tutorials/feature-use/' + feature_tutorial[:-6] + '.nblink"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))

# get list of tutorial files inside folder application-example
app_tutorial_list = os.listdir("./tutorials/application-example")
for app_tutorial in app_tutorial_list:
    if app_tutorial.endswith(".ipynb"):
        # iterate through application-example tutorial files and create nb-link file
        with open("./docs/source/example_links/" + app_tutorial[:-6] + ".nblink", "w") as out:
            line1 = "{"
            line2 = '    "path": "../../../tutorials/application-example/' + app_tutorial[:-6] + '.nblink"'
            line3 = "}"
            out.write("{}\n{}\n{}\n".format(line1, line2, line3))
