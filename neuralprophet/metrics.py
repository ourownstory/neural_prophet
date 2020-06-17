from abc import abstractmethod
import torch


class Metric:
    """Base class for all Metrics."""
    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name
        self._sum = 0
        self._num_examples = 0
        self.stored_values = []
        self.total_updates = 0

    def reset(self):
        """Resets the metric to it's initial state.

        By default, this is called at the start of each epoch.
        """
        self._sum = 0
        self._num_examples = 0

    @abstractmethod
    def update(self, predicted, target):
        """Updates the metric's state using the passed batch output.

        By default, this is called once for each batch.
        Args:
            predicted: the output from the model's forward function.
            target: actual values
        """
        self.total_updates += 1
        pass

    def compute(self, save=False):
        if self._num_examples == 0: self.no_sample_error()
        value = self._sum / self._num_examples
        # value = value.data.item()
        if save: self.stored_values.append(value)
        return value

    def no_sample_error(self):
        raise ValueError(self.name, " must have at least one example before it can be computed.")

    def __str__(self):
        return "{}: {:8.3f}".format(self.name, self.compute())

    def print_stored(self):
        print("{}: ".format(self.name))
        print("; ".join(["{:6.3f}".format(x) for x in self.stored_values]))


class BatchMetric(Metric):
    """Calculates a metric from batch model predictions."""

    def __init__(self, name=None, specific_column=None):
        """
        Args:
            name (str): Metric name, if not same as cls name
            specific_column (int): compute metric only over this column of the model outputs.
        """
        super(BatchMetric, self).__init__(name)
        if specific_column is not None:
            self.name = "{}-{}".format(self.name, str(specific_column+1))
        self.specific_column = specific_column

    def update(self, predicted, target, **kwargs):
        """Updates the metric's state using the passed batch output.

        By default, this is called once for each batch.
        Args:
            predicted: the output from the model's forward function.
            target: actual values
            kwargs: passed on to function that computes the metric.
        """
        self.total_updates += 1
        num = target.shape[0]
        if self.specific_column is not None:
            predicted = predicted[:, self.specific_column]
            target = target[:, self.specific_column]
        avg_value = self._batch_value(predicted, target, **kwargs)
        self._sum += avg_value * num
        self._num_examples += num

    @abstractmethod
    def _batch_value(self, predicted, target, **kwargs):
        """Computes the metrics avg value over the batch.

            Called inside update()
        Args:
            predicted: the output from the model's forward function.
            target: actual values
        """
        pass


class MAE(BatchMetric):
    """Calculates the mean absolute error."""
    def __init__(self, specific_column=None):
        super(MAE, self).__init__(specific_column=specific_column)

    def _batch_value(self, predicted, target, **kwargs):
        # absolute_errors = torch.abs(predicted - target.view_as(predicted))
        absolute_errors = torch.abs(predicted - target)
        return torch.mean(absolute_errors).data.item()


class MSE(BatchMetric):
    """Calculates the mean squared error."""
    def __init__(self, specific_column=None):
        super(MSE, self).__init__(specific_column=specific_column)

    def _batch_value(self, predicted, target, **kwargs):
        # squared_errors = torch.pow(predicted - target.view_as(predicted), 2)
        squared_errors = torch.pow(predicted - target, 2)
        return torch.mean(squared_errors).data.item()


class Loss(BatchMetric):
    """Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target tensor, optionally other arguments,
            and returns the average loss over all observations in the batch.
    """
    def __init__(self, loss_fn, specific_column=None):
        super(Loss, self).__init__(
            name=loss_fn.__class__.__name__,
            specific_column=specific_column)
        self._loss_fn = loss_fn


    def _batch_value(self, predicted, target, **kwargs):
        average_loss = self._loss_fn(predicted, target, **kwargs)
        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")
        return average_loss.data.item()


class Value(Metric):
    """Keeps track of a value as a metric."""
    def __init__(self, name):
        super(Value, self).__init__(name=name)

    def update(self, avg_value, num):
        self.total_updates += 1
        self._sum += avg_value.data.item() * num
        self._num_examples += num

