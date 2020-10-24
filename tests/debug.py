
if __name__ == '__main__':
    # if called directly
    # TODO: add argparse to allow for plotting with tests using command line
    # TODO: add hard performance criteria to training tests, setting seeds

    # uncomment to run tests with plotting or debug logs print output and  respectively

    # default option
    UnitTests.plot = False
    log.setLevel("DEBUG")
    log.parent.setLevel("WARNING")

    # not verbose option
    # UnitTests.plot = False
    # log.setLevel("ERROR")
    # log.parent.setLevel("ERROR")

    # very verbose option
    # UnitTests.plot = True
    # log.setLevel("DEBUG")
    # log.parent.setLevel("DEBUG")
    print("HELLO")
    # tests = UnitTests()

    # to run all tests
    # unittest.main(exit=False)

    # to run individual tests
    # tests.test_impute_missing()
    # tests.test_time_dataset()




