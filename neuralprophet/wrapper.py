from neuralprophet.forecaster import NeuralProphet


class Prophet(NeuralProphet):
    def __init__(self, *args, **kwargs):
        super(Prophet, self).__init__(*args, **kwargs)
