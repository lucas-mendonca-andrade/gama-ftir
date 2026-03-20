import pytest
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.preprocessing import StandardScaler

from gama.configuration.parser import merge_configurations, pset_from_config


def test_merge_configuration():
    """Test merging two simple configurations works as expected."""

    one = {"alpha": [0, 1], BernoulliNB: {"fit_prior": [True, False]}}
    two = {"alpha": [0, 2], GaussianNB: {"fit_prior": [True, False]}}
    expected_merged = {
        "alpha": [0, 1, 2],
        GaussianNB: {"fit_prior": [True, False]},
        BernoulliNB: {"fit_prior": [True, False]},
    }

    actual_merged = merge_configurations(one, two)
    assert expected_merged == actual_merged


def test_mandatory_primitive_parsed():
    cfg = {
        GaussianNB: {"mandatory": [True]},
        StandardScaler: {},
    }
    _, _, mandatory = pset_from_config(cfg)
    assert mandatory == frozenset({"GaussianNB"})


def test_mandatory_invalid_value_raises():
    cfg = {GaussianNB: {"mandatory": [False]}}
    with pytest.raises(ValueError, match="mandatory"):
        pset_from_config(cfg)
