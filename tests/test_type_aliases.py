from typing import Dict, List

import pytest

from stable_baselines3.common.type_aliases import check_cast, non_null


def test_non_null():
    for a in (1, "a", [2]):
        assert non_null(a) == a

    with pytest.raises(ValueError):
        non_null(None)


def test_check_cast():
    EMPTY_DICT = {}
    assert check_cast(dict, EMPTY_DICT) is EMPTY_DICT
    assert check_cast(dict[str, int], EMPTY_DICT) is EMPTY_DICT
    assert check_cast(Dict[str, int], EMPTY_DICT) is EMPTY_DICT

    with pytest.raises(TypeError):
        check_cast(list[int], EMPTY_DICT)
        check_cast(List[int], EMPTY_DICT)

    # NOTE: check_cast does not check the template arguments, only the main class.
    # Tests should give an accurate understanding of how the function works, so we still check for this behavior.
    a: list[str] = ["a"]
    assert (
        check_cast(list[int], a) is a
    ), "If you managed to write code to trigger this assert that's good! We'd like template arguments to be checked."
