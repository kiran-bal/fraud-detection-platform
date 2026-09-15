import pandas as pd
import pytest

from fraud_platform.data.schema import ValidationError, validate, validate_or_raise


def test_valid_frame_passes(transactions):
    report = validate(transactions)
    assert report.ok and report.rows == len(transactions)


def test_missing_column_is_an_error(transactions):
    report = validate(transactions.drop(columns=["V7"]))
    assert not report.ok and "missing columns" in report.errors[0]


def test_nulls_negative_amounts_and_duplicates(transactions):
    bad = transactions.copy()
    bad.loc[0, "V1"] = None
    bad.loc[1, "amount"] = -5
    bad.loc[2, "transaction_id"] = bad.loc[3, "transaction_id"]
    report = validate(bad)
    assert len(report.errors) == 3


def test_scoring_schema_does_not_need_label(transactions):
    assert validate(transactions.drop(columns=["is_fraud"]), training=False).ok
    assert not validate(transactions.drop(columns=["is_fraud"]), training=True).ok


def test_no_positives_is_an_error(transactions):
    df = transactions.assign(is_fraud=0)
    with pytest.raises(ValidationError):
        validate_or_raise(df)


def test_unsorted_time_is_only_a_warning(transactions):
    report = validate(transactions.sample(frac=1, random_state=1))
    assert report.ok and any("not sorted" in w for w in report.warnings)


def test_empty_frame():
    assert not validate(pd.DataFrame(columns=["transaction_id"])).ok
