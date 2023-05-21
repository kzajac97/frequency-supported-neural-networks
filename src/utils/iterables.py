from typing import Any, Optional


def filter_dict(function: Optional[callable], d: dict) -> dict:
    """Filters dict based on values"""

    def default_function(value: Any) -> bool:
        return value is not None

    if function is None:
        function = default_function

    return {
        key: value for key, value in d.items() if function(value)
    }  # TODO: Invert logic to align with standard `filter`


def collect_keys_with_prefix(parameters: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Returns subset of dict key-value pairs with keys starting with corresponding prefix"""
    return {key: value for key, value in parameters.items() if key.startswith(prefix)}
