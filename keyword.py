"""CLI wrapper for the custom Naver DataLab keyword runner.

The file name is intentionally kept as ``keyword.py`` for manual execution.
When Python imports ``keyword`` as a standard-library module from this working
directory, this wrapper exposes the small stdlib-compatible API instead of
loading the project runner.
"""

from __future__ import annotations

if __name__ == "keyword":
    kwlist = [
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    ]
    softkwlist = ["_", "case", "match", "type"]
    iskeyword = frozenset(kwlist).__contains__
    issoftkeyword = frozenset(softkwlist).__contains__
else:
    from src.crawler.sources.keyword_sector_runner import main, run, run_scheduled

    __all__ = ["main", "run", "run_scheduled"]

    if __name__ == "__main__":
        main()
