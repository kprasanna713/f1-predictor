"""Test the Ergast time parser, no network calls."""
from f1_predictor.ergast_client import _parse_time


def test_parse_minutes_seconds():
    assert _parse_time("1:23.456") == 60 + 23.456


def test_parse_seconds_only():
    assert _parse_time("85.123") == 85.123


def test_parse_empty():
    assert _parse_time("") is None


def test_parse_garbage():
    assert _parse_time("not a time") is None
