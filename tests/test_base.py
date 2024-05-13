from dataclasses import fields

from src.base import TestParams


def test_params_fields():
    fields_name = [field.name for field in fields(TestParams)]
    expected_fields = ["test_statistic", "test_pval", "test_h0", "is_fitted"]
    assert all(
        var in fields_name for var in expected_fields
    ), "Dataclass is missing expected variables"
