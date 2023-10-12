from typing import Dict, List

import pytest

from stable_baselines3.common.type_aliases import check_cast, non_null


def test_non_null():
    for a in (1, "a", [2]):
        assert non_null(a) == a

    with pytest.raises(ValueError):
        non_null(None)


def test_check_cast():
    assert check_cast(dict, {}) == {}
    assert check_cast(dict[str, int], {}) == {}
    assert check_cast(Dict[str, int], {}) == {}

    with pytest.raises(TypeError):
        check_cast(list[int], {})
        check_cast(List[int], {})

    # NOTE: check_cast does not check the template arguments, only the main class.
    a: list[str] = ["a"]
    assert check_cast(list[int], a) == a
