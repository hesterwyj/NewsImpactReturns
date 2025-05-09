"""
Demo for using data iterator with Quantile DMatrix
==================================================

    .. versionadded:: 1.2.0

The demo that defines a customized iterator for passing batches of data into
:py:class:`xgboost.QuantileDMatrix` and use this ``QuantileDMatrix`` for training.  The
feature is primarily designed to reduce the required GPU memory for training on
distributed environment.

Aftering going through the demo, one might ask why don't we use more native Python
iterator?  That's because XGBoost requires a `reset` function, while using
`itertools.tee` might incur significant memory usage according to:

  https://docs.python.org/3/library/itertools.html#itertools.tee.

.. seealso::

  :ref:`sphx_glr_python_examples_external_memory.py`

"""

from typing import Callable

import cupy
import numpy

import xgboost

COLS = 64
ROWS_PER_BATCH = 1000  # data is splited by rows
BATCHES = 32


class IterForDMatrixDemo(xgboost.core.DataIter):
    """A data iterator for XGBoost DMatrix.

    `reset` and `next` are required for any data iterator, other functions here
    are utilites for demonstration's purpose.

    """

    def __init__(self) -> None:
        """Generate some random data for demostration.

        Actual data can be anything that is currently supported by XGBoost.
        """
        self.rows = ROWS_PER_BATCH
        self.cols = COLS
        rng = cupy.random.RandomState(numpy.uint64(1994))
        self._data = [rng.randn(self.rows, self.cols)] * BATCHES
        self._labels = [rng.randn(self.rows)] * BATCHES
        self._weights = [rng.uniform(size=self.rows)] * BATCHES

        self.it = 0  # set iterator to 0
        super().__init__()

    def as_array(self) -> cupy.ndarray:
        return cupy.concatenate(self._data)

    def as_array_labels(self) -> cupy.ndarray:
        return cupy.concatenate(self._labels)

    def as_array_weights(self) -> cupy.ndarray:
        return cupy.concatenate(self._weights)

    def data(self) -> cupy.ndarray:
        """Utility function for obtaining current batch of data."""
        return self._data[self.it]

    def labels(self) -> cupy.ndarray:
        """Utility function for obtaining current batch of label."""
        return self._labels[self.it]

    def weights(self) -> cupy.ndarray:
        return self._weights[self.it]

    def reset(self) -> None:
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data: Callable) -> bool:
        """Yield the next batch of data."""
        if self.it == len(self._data):
            # Return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        input_data(data=self.data(), label=self.labels(), weight=self.weights())
        self.it += 1
        return True


def main() -> None:
    rounds = 100
    it = IterForDMatrixDemo()

    # Use iterator, must be `QuantileDMatrix`.

    # In this demo, the input batches are created using cupy, and the data processing
    # (quantile sketching) will be performed on GPU. If data is loaded with CPU based
    # data structures like numpy or pandas, then the processing step will be performed
    # on CPU instead.
    m_with_it = xgboost.QuantileDMatrix(it)

    # Use regular DMatrix.
    m = xgboost.DMatrix(
        it.as_array(), it.as_array_labels(), weight=it.as_array_weights()
    )

    assert m_with_it.num_col() == m.num_col()
    assert m_with_it.num_row() == m.num_row()
    # Tree method must be `hist`.
    reg_with_it = xgboost.train(
        {"tree_method": "hist", "device": "cuda"},
        m_with_it,
        num_boost_round=rounds,
        evals=[(m_with_it, "Train")],
    )
    predict_with_it = reg_with_it.predict(m_with_it)

    reg = xgboost.train(
        {"tree_method": "hist", "device": "cuda"},
        m,
        num_boost_round=rounds,
        evals=[(m, "Train")],
    )
    predict = reg.predict(m)


if __name__ == "__main__":
    main()
