"""Arrow-key terminal components following ominf data_process's interaction style."""

# Chinese UI punctuation is intentional.
# ruff: noqa: RUF001
import questionary

STYLE = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:ansiblack"),
        ("instruction", "fg:#888888"),
        ("text", ""),
        ("disabled", "fg:#858585 italic"),
    ]
)


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def select(title, options, default=None):
    choices = [questionary.Choice(label, value=value) for value, label in options]
    selected = next((c for c in choices if c.value == default), None)
    return questionary.select(
        title, choices=choices, default=selected, style=STYLE, instruction="(↑↓ 选择，Enter 确认，Ctrl+C 返回/退出)"
    ).ask()


def checkbox(title, options, defaults):
    choices = [questionary.Choice(label, value=value, checked=value in defaults) for value, label in options]
    return questionary.checkbox(
        title, choices=choices, style=STYLE, instruction="(↑↓ 移动，Space 勾选，Enter 确认，Ctrl+C 返回)"
    ).ask()


def text(title, default="", validate=None):
    return questionary.text(title, default=default, validate=validate, style=STYLE).ask()


def path(title, default=""):
    return questionary.path(title, default=default, style=STYLE).ask()


def confirm(title, default):
    return questionary.confirm(title, default=default, auto_enter=False, style=STYLE).ask()
