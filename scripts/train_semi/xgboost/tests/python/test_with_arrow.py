import os

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.core import DataSplitMode

pytestmark = pytest.mark.skipif(
    tm.no_arrow()["condition"] or tm.no_pandas()["condition"],
    reason=tm.no_arrow()["reason"] + " or " + tm.no_pandas()["reason"],
)

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc


class TestArrowTable:
    def test_arrow_table(self):
        df = pd.DataFrame(
            [[0, 1, 2.0, 3.0], [1, 2, 3.0, 4.0]], columns=["a", "b", "c", "d"]
        )
        table = pa.Table.from_pandas(df)
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 2
        assert dm.num_col() == 4

    def test_arrow_table_with_label(self):
        df = pd.DataFrame([[1, 2.0, 3.0], [2, 3.0, 4.0]], columns=["a", "b", "c"])
        table = pa.Table.from_pandas(df)
        label = np.array([0, 1])
        dm = xgb.DMatrix(table)
        dm.set_label(label)
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([0, 1]))

    def test_arrow_table_from_np(self):
        coldata = np.array(
            [[1.0, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]]
        )
        cols = list(map(pa.array, coldata))
        table = pa.Table.from_arrays(cols, ["a", "b", "c"])
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 4
        assert dm.num_col() == 3

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_arrow_train(self, DMatrixT):
        import pandas as pd

        rows = 100
        X = pd.DataFrame(
            {
                "A": np.random.randint(0, 10, size=rows),
                "B": np.random.randn(rows),
                "C": np.random.permutation([1, 0] * (rows // 2)),
            }
        )
        y = pd.Series(np.random.randn(rows))

        table = pa.Table.from_pandas(X)
        dtrain1 = DMatrixT(table)
        dtrain1.set_label(pa.Table.from_pandas(pd.DataFrame(y)))
        bst1 = xgb.train({}, dtrain1, num_boost_round=10)
        preds1 = bst1.predict(DMatrixT(X))

        dtrain2 = DMatrixT(X, y)
        bst2 = xgb.train({}, dtrain2, num_boost_round=10)
        preds2 = bst2.predict(DMatrixT(X))

        np.testing.assert_allclose(preds1, preds2)

        preds3 = bst2.inplace_predict(table)
        np.testing.assert_allclose(preds1, preds3)
        assert bst2.feature_names == ["A", "B", "C"]
        assert bst2.feature_types == ["int", "float", "int"]

    def test_arrow_survival(self):
        data = os.path.join(tm.data_dir(__file__), "veterans_lung_cancer.csv")
        table = pc.read_csv(data)
        y_lower_bound = table["Survival_label_lower_bound"]
        y_upper_bound = table["Survival_label_upper_bound"]
        X = table.drop(["Survival_label_lower_bound", "Survival_label_upper_bound"])

        dtrain = xgb.DMatrix(
            X, label_lower_bound=y_lower_bound, label_upper_bound=y_upper_bound
        )
        y_np_up = dtrain.get_float_info("label_upper_bound")
        y_np_low = dtrain.get_float_info("label_lower_bound")
        np.testing.assert_equal(y_np_up, y_upper_bound.to_pandas().values)
        np.testing.assert_equal(y_np_low, y_lower_bound.to_pandas().values)


@pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
class TestArrowTableColumnSplit:
    def test_arrow_table(self):
        def verify_arrow_table():
            df = pd.DataFrame(
                [[0, 1, 2.0, 3.0], [1, 2, 3.0, 4.0]], columns=["a", "b", "c", "d"]
            )
            table = pa.Table.from_pandas(df)
            dm = xgb.DMatrix(table, data_split_mode=DataSplitMode.COL)
            assert dm.num_row() == 2
            assert dm.num_col() == 4 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_arrow_table)
