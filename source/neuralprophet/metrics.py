from abc import abstractmethod
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("NP.metrics")


class MetricsCollection:
    """Collection of Metrics that performs action over all"""

    def __init__(self, metrics, value_metrics=None):

        self.batch_metrics = []
        self.value_metrics = OrderedDict({})
        for m in metrics:
            if isinstance(m, BatchMetric):
                self.batch_metrics.append(m)
            else:
                raise ValueError("Metric {} not BatchMetric".format(m._class__.__name__))
        if value_metrics is not None:
            for vm in value_metrics:
                if isinstance(vm, ValueMetric):
                    self.value_metrics[vm.name] = vm
                else:
                    raise ValueError("Metric {} not ValueMetric".format(vm._class__.__name__))

    @property
    def total_updates(self):
        return self.batch_metrics[0].total_updates

    @property
    def all(self):
        return self.batch_metrics + list(self.value_metrics.values())

    def reset(self, hard=False):
        """reset all"""
        for m in self.all:
            m.reset(hard=hard)

    def update_batch(self, predicted, target):
        """update BatchMetrics"""
        for m in self.batch_metrics:
            m.update(predicted=predicted, target=target)

    def update_values(self, values, num):
        """update ValueMetrics.

        Args:
            values (dict): dict with matching names (to defined ValueMetrics),
                containing average values over batch/update step
            num (int): number of samples in batch/update step
        """
        for name, value in values.items():
            if name in self.value_metrics.keys():
                self.value_metrics[name].update(avg_value=value, num=num)
        not_updated = set(self.value_metrics.keys()) - set(values.keys())
        if len(not_updated) > 0:
            raise ValueError("Metrics {} defined but not updated.".format(not_updated))

    def update(self, predicted, target, values=None):
        """update all metrics.

        Args:
            predicted: the output from the model's forward function.
            target: actual values
            values (dict): dict with matching names to defined ValueMetrics
                Note: if the correct name is not supplied, the metric is not updated.
        """
        self.update_batch(predicted=predicted, target=target)
        if values is not None:
            self.update_values(values=values, num=target.shape[0])

    def compute(self, save=False):
        """calculates the current value of the metric

        Args:
            save (bool): whether to add the current value to stored_values
        Returns:
            dict of current values of all metrics
        """
        metrics = OrderedDict({})
        for m in self.all:
            metrics[m.name] = m.compute(save=save)
        return metrics

    def get_stored(self, loc=None):
        """Creates an OrderedDict from stored metric values

        Args:
            loc (int): if only stored value at this location to be retrieved

        Returns:
            OrderedDict
        """
        metrics = OrderedDict({})
        for m in self.all:
            if loc is None:
                metrics[m.name] = m.stored_values
            else:
                metrics[m.name] = [m.stored_values[loc]]
        return metrics

    def get_stored_as_df(self, loc=None):
        """Creates an Dataframe from stored metric values

        Args:
            loc (int): if only stored value at this location to be retrieved

        Returns:
            pd.Dataframe
        """
        metrics = pd.DataFrame(self.get_stored(loc=loc))
        return metrics

    def add_specific_target(self, target_pos):
        """Duplicates BatchMetrics with their version for a specific target.

        Args:
            target_pos (int, list): index of target to compute metrics over
        """
        specific_metrics = []
        if isinstance(target_pos, int):
            target_pos = [target_pos]
        for pos in target_pos:
            for m in self.batch_metrics:
                sm = m.new(specific_column=pos)
                specific_metrics.append(sm)
        self.batch_metrics.extend(specific_metrics)

    def set_shift_scale(self, shift_scale):
        """Adds data denormalization params to applicable metrics

        Args:
            shift_scale (tuple, float): data shift and scale parameters
        """
        for m in self.all:
            m.set_shift_scale(shift_scale)

    def __str__(self):
        """Nice-prints current values"""
        metrics_string = pd.DataFrame({**self.compute()}, index=[0]).to_string(
            float_format=lambda x: "{:6.3f}".format(x)
        )
        return metrics_string

    def print(self, loc=None):
        """Nice-prints stored values"""
        metrics_string = self.get_stored_as_df(loc=loc).to_string(float_format=lambda x: "{:6.3f}".format(x))
        log.debug(metrics_string)


class Metric:
    """Base class for all Metrics."""

    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name
        self._sum = 0
        self._num_examples = 0
        self.stored_values = []
        self.total_updates = 0

    def reset(self, hard=False):
        """Resets the metric to it's initial state.

        By default, this is called at the start of each epoch.
        """
        self._sum = 0
        self._num_examples = 0
        if hard:
            self.stored_values = []
            self.total_updates = 0

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
        """calculates the current value of the metric

        Args:
            save (bool): whether to add the current value to stored_values

        Returns:
            current value of the metric
        """
        if self._num_examples == 0:
            self._no_sample_error()
        value = self._sum / self._num_examples
        # value = value.data.item()
        if save:
            self.stored_values.append(value)
        return value

    def _no_sample_error(self):
        raise ValueError(self.name, " must have at least one example before it can be computed.")

    def __str__(self):
        """Nice-prints current value"""
        return "{}: {:8.3f}".format(self.name, self.compute())

    def print_stored(self):
        """Nice-prints stored values"""
        log.debug("{}: ".format(self.name))
        log.debug("; ".join(["{:6.3f}".format(x) for x in self.stored_values]))

    def set_shift_scale(self, shift_scale):
        """placeholder for subclasses to implement if applicable"""
        pass

    def new(self):
        return self.__class__(name=self.name)


class BatchMetric(Metric):
    """Calculates a metric from batch model predictions."""

    def __init__(self, name=None, specific_column=None):
        """
        Args:
            name (str): Metric name, if not same as cls name
            specific_column (int): compute metric only over this column of the model outputs.
            shift_scale (tuple, float): data shift and scale parameters
        """
        super(BatchMetric, self).__init__(name)
        if specific_column is not None:
            self.name = "{}-{}".format(self.name, str(specific_column + 1))
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
        avg_value = self._update_batch_value(predicted, target, **kwargs)
        self._sum += avg_value * num
        self._num_examples += num

    @abstractmethod
    def _update_batch_value(self, predicted, target, **kwargs):
        """Computes the metrics avg value over the batch.

            Called inside update()
        Args:
            predicted: the output from the model's forward function.
            target: actual values
        """
        pass

    @abstractmethod
    def new(self, specific_column=None):
        """

        Args:
            specific_column (int): calculate metric only over target at pos

        Returns:
            copy of metric instance, reset
        """
        if specific_column is None and self.specific_column is not None:
            specific_column = self.specific_column
        new_cls = self.__class__(specific_column=specific_column)
        return new_cls


class MAE(BatchMetric):
    """Calculates the mean absolute error."""

    def __init__(self, specific_column=None, shift_scale=None):
        super(MAE, self).__init__(specific_column=specific_column)
        self.shift_scale = shift_scale

    def _update_batch_value(self, predicted, target, **kwargs):
        predicted = predicted.numpy()
        target = target.numpy()
        if self.shift_scale is not None:
            predicted = self.shift_scale[1] * predicted + self.shift_scale[0]
            target = self.shift_scale[1] * target + self.shift_scale[0]
        absolute_errors = np.abs(predicted - target)
        return np.mean(absolute_errors)

    def set_shift_scale(self, shift_scale):
        """Adds data denormalization params

        Args:
            shift_scale (tuple, float): data shift and scale parameters
        """
        self.shift_scale = shift_scale

    def new(self, specific_column=None, shift_scale=None):
        """

        Args:
            specific_column (int): calculate metric only over target at pos
            shift_scale (tuple, float): data shift and scale parameters

        Returns:
            copy of metric instance, reset
        """
        if specific_column is None and self.specific_column is not None:
            specific_column = self.specific_column
        if shift_scale is None and self.shift_scale is not None:
            shift_scale = self.shift_scale
        return self.__class__(specific_column=specific_column, shift_scale=shift_scale)


class MSE(BatchMetric):
    """Calculates the mean squared error."""

    def __init__(self, specific_column=None, shift_scale=None):
        super(MSE, self).__init__(specific_column=specific_column)
        self.shift_scale = shift_scale

    def _update_batch_value(self, predicted, target, **kwargs):
        predicted = predicted.numpy()
        target = target.numpy()
        if self.shift_scale is not None:
            predicted = self.shift_scale[1] * predicted + self.shift_scale[0]
            target = self.shift_scale[1] * target + self.shift_scale[0]
        squared_errors = (predicted - target) ** 2
        return np.mean(squared_errors)

    def set_shift_scale(self, shift_scale):
        """Adds data denormalization params.

        Args:
            shift_scale (tuple, float): data shift and scale parameters
        """
        self.shift_scale = shift_scale

    def new(self, specific_column=None, shift_scale=None):
        """

        Args:
            specific_column (int): calculate metric only over target at pos
            shift_scale (tuple, float): data shift and scale parameters

        Returns:
            copy of metric instance, reset
        """
        if specific_column is None and self.specific_column is not None:
            specific_column = self.specific_column
        if shift_scale is None and self.shift_scale is not None:
            shift_scale = self.shift_scale
        return self.__class__(specific_column=specific_column, shift_scale=shift_scale)


class LossMetric(BatchMetric):
    """Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target tensor, optionally other arguments,
            and returns the average loss over all observations in the batch.
    """

    def __init__(self, loss_fn, specific_column=None):
        super(LossMetric, self).__init__(name=loss_fn.__class__.__name__, specific_column=specific_column)
        self._loss_fn = loss_fn

    def _update_batch_value(self, predicted, target, **kwargs):
        average_loss = self._loss_fn(predicted, target, **kwargs)
        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")
        return average_loss.data.item()

    def new(self, specific_column=None):
        if specific_column is None and self.specific_column is not None:
            specific_column = self.specific_column
        return self.__class__(loss_fn=self._loss_fn, specific_column=specific_column)


class ValueMetric(Metric):
    """Keeps track of a value as a metric."""

    def __init__(self, name):
        super(ValueMetric, self).__init__(name=name)

    def update(self, avg_value, num):
        """

        Args:
            avg_value (float): average value over batch/update step
            num (int): number of samples in batch/update step
        """
        self.total_updates += 1
        self._sum += avg_value.data.item() * num
        self._num_examples += num
