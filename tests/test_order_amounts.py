import pandas as pd

from datamate.helpers import order_amounts


def test_line_amounts_are_summed():
    df = pd.DataFrame({"id": ["A", "A", "A", "B"], "tien": ["1,000", "2,000", "3,000", "5k"]})
    out = order_amounts(df, "id", "tien").set_index("id")["amount"]
    assert out["A"] == 6000 and out["B"] == 5000      # the failure seen on the real preflight: drop_duplicates would give 1000


def test_order_totals_taken_once():
    df = pd.DataFrame({"id": ["A", "A", "B", "B", "B"], "tong": [6000, 6000, 9000, 9000, 9000]})
    out = order_amounts(df, "id", "tong").set_index("id")["amount"]
    assert out["A"] == 6000 and out["B"] == 9000
