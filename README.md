# Bifrost Project presents: NeuralProphet
A Neural Network based Time-Series model, heavily inspired by [Facebook Prophet](https://github.com/facebook/prophet).

## Install
After downloading the code (manually or via `git clone`), install neuralprophet as python package with
`cd bifrost`
`pip install [-e] .`

Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

Now in any notebook you can do:

`import neuralprophet`

Or

`from neuralprophet.neural_prophet import NeuralProphet`


## Contribute
As far as possible, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.
