import numpy as np
import types
from keras import Sequential
from keras.utils.generic_utils import has_arg
from keras.wrappers.scikit_learn import BaseWrapper
import copy
from keras.utils.np_utils import to_categorical


class FitGeneratorBaseWrapper(BaseWrapper):

    def __init__(self, build_fn, generator, **sk_params):
        super(FitGeneratorBaseWrapper, self).__init__(build_fn, **sk_params)
        self.generator = generator

    def check_params(self, params):
        """Checks for user typos in `params`.
                # Arguments
                    params: dictionary; the parameters to be checked
                # Raises
                    ValueError: if any member of `params` is not a valid argument.
                """
        legal_params_fns = [Sequential.fit, Sequential.fit_generator, Sequential.predict,
                            Sequential.predict_classes, Sequential.evaluate]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)

        if self.generator is not None:
            legal_params_fns.append(self.generator)

        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    break
                else:
                    if params_name != 'nb_epoch':
                        raise ValueError(
                            '{} is not a legal parameter'.format(params_name))

    def fit(self, x, y, **kwargs):
        return self.fit_generator(x, y, **kwargs)

    def fit_generator(self, x, y, **kwargs):
        """
        Constructs a new model with `build_fn` & fit_generator the model with generator data from `(x, y)`
        :param kwargs:  dictionary arguments, legal arguments of `Sequential.fit_generator`
        :return: details on the training history at each epochs
        """

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_generator_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        generator = self.generator(x, y, **fit_generator_args)
        history = self.model.fit_generator(generator, **kwargs)

        return history


class FitGeneratorKerasClassifier(FitGeneratorBaseWrapper):
    def fit(self, x, y, **kwargs):
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        return super(FitGeneratorKerasClassifier, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        classes = self.model.predict_classes(x, **kwargs)
        return self.classes_[classes]
