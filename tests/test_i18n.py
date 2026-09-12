"""Language resolution and the shape of the localized string tables."""

from __future__ import annotations

import pytest

from ollama_mcp import config, i18n


def test_explicit_settings_win_over_everything() -> None:
    assert i18n.resolve("ja", hint="purely english text") is i18n.Language.JA
    assert i18n.resolve("en", hint="完全に日本語のテキスト") is i18n.Language.EN


def test_auto_follows_the_task_text() -> None:
    assert i18n.resolve(i18n.AUTO, hint="docstring を日本語で追加して") is i18n.Language.JA
    assert i18n.resolve(i18n.AUTO, hint="add google-style docstrings") is i18n.Language.EN


def test_auto_falls_back_to_the_machine_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LC_ALL", "LC_MESSAGES", "LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "ja_JP.UTF-8")
    assert i18n.resolve(i18n.AUTO) is i18n.Language.JA
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setattr(i18n, "detect_from_locale", lambda: None)
    assert i18n.resolve(i18n.AUTO) is i18n.Language.EN


def test_a_locale_merely_starting_with_ja_is_not_japanese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "jam_JM.UTF-8")
    monkeypatch.setattr(i18n.locale, "getlocale", lambda *a: (None, None))
    assert i18n.detect_from_locale() is None


@pytest.mark.parametrize("value", ["ja", "JA", " en ", "auto"])
def test_valid_settings_normalize(value: str) -> None:
    assert i18n.normalize_setting(value) == value.strip().lower()


def test_unknown_language_is_rejected_with_the_allowed_set_named() -> None:
    with pytest.raises(ValueError, match="auto, en, ja"):
        i18n.normalize_setting("klingon")


def test_non_string_language_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        i18n.normalize_setting(7)


def test_every_language_has_a_complete_string_table() -> None:
    for language in i18n.Language:
        table = i18n.strings(language)
        for field_name in table.__dataclass_fields__:
            assert getattr(table, field_name), f"{language.value}.{field_name} is empty"


def test_both_prompt_pairs_demand_the_ascii_keywords() -> None:
    # A localized DONE:/ESCALATE: keyword would break every published CLAUDE.md
    # delegation policy, so both prompts have to insist on the ASCII form.
    for language in i18n.Language:
        table = i18n.strings(language)
        for prompt in (table.edit_system, table.read_system):
            assert "DONE:" in prompt
            assert "ESCALATE:" in prompt


def test_read_prompt_accepts_the_budget_placeholder() -> None:
    for language in i18n.Language:
        assert "1500" in i18n.strings(language).read_system.format(budget=1500)


def test_config_reads_and_validates_the_language(tmp_path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text('[i18n]\nlanguage = "ja"\n', encoding="utf-8")
    assert config.load(tmp_path).i18n.language == "ja"


def test_config_defaults_to_auto(tmp_path) -> None:
    assert config.load(tmp_path).i18n.language == i18n.AUTO


def test_env_overrides_the_config_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text('[i18n]\nlanguage = "en"\n', encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MCP_LANG", "ja")
    assert config.load(tmp_path).i18n.language == "ja"


def test_bad_language_fails_at_load_not_at_first_delegation(tmp_path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text('[i18n]\nlanguage = "jp"\n', encoding="utf-8")
    with pytest.raises(config.ConfigError, match="jp"):
        config.load(tmp_path)


def test_a_bom_does_not_break_the_config_file(tmp_path) -> None:
    # What a Japanese Windows editor writes by default.
    (tmp_path / ".ollama-mcp.toml").write_bytes(
        b"\xef\xbb\xbf" + b'[i18n]\nlanguage = "ja"\n'
    )
    assert config.load(tmp_path).i18n.language == "ja"


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text("[gate]\ntimeuot_s = 5\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="timeuot_s"):
        config.load(tmp_path)


def test_wrong_type_is_rejected(tmp_path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text(
        "[limits]\nmax_iterations = true\n", encoding="utf-8"
    )
    with pytest.raises(config.ConfigError, match="must be int"):
        config.load(tmp_path)


def test_an_int_is_accepted_where_a_float_is_expected(tmp_path) -> None:
    (tmp_path / ".ollama-mcp.toml").write_text("[models]\ntemperature = 0\n", encoding="utf-8")
    assert config.load(tmp_path).models.temperature == 0.0
