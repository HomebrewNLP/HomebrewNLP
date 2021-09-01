from rich import print
from rich.console import Console
from rich.syntax import Syntax
from typing import Optional
from rich.traceback import install


# Color coded tracebacks
# install(show_locals=False)

console = Console()


# TODO: Allow for users to choose theme
def syntax_print(string: str, language: Optional[str] = "python", theme: Optional[str] = "monokai",
                 title: Optional[str] = None) -> None:
    if title is not None:
        console.rule(title)
    syntax = Syntax(string, language, theme=theme, line_numbers=True)
    console.print(syntax)


def pretty_print(*data):
    print(*data)


def log(*data, locals: bool = False):
    console.log(*data, log_locals=locals)
