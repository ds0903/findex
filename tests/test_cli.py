import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from findex.cli import app

runner = CliRunner()


def run(*args: str | Path) -> Result:
    # COLUMNS, щоб rich не переносив слова в таблиці на вузькому "терміналі" тестів
    return runner.invoke(app, [str(a) for a in args], env={"COLUMNS": "200"})


@pytest.fixture
def built(corpus: Path, tmp_path: Path) -> Path:
    out = tmp_path / "index.bin"
    result = run("index", corpus, "--out", out, "--no-memory", "--positions")
    assert result.exit_code == 0, result.output
    return out


# --- загальне ---


def test_help_lists_commands():
    result = run("--help")
    assert result.exit_code == 0
    for name in ("index", "search", "stats"):
        assert name in result.stdout


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "index" in result.output


def test_version():
    result = run("--version")
    assert result.exit_code == 0
    assert result.stdout.startswith("findex ")


# --- index ---


@pytest.mark.parametrize("fmt", ["binary", "pickle"])
def test_index_then_search(corpus: Path, tmp_path: Path, fmt: str):
    out = tmp_path / "index.bin"
    result = run("index", corpus, "--out", out, "--format", fmt)
    assert result.exit_code == 0, result.output
    assert "документів:   2" in result.stdout
    assert "пік пам'яті" in result.stdout
    assert out.exists()

    result = run("search", out, "sea NOT whale", "--no-snippets")
    assert result.exit_code == 0, result.output
    assert "знайдено документів: 1" in result.stdout
    assert "Ship Book" in result.stdout
    assert "Whale Book" not in result.stdout


def test_index_options(corpus: Path, tmp_path: Path):
    out = tmp_path / "index.bin"
    result = run("index", corpus, "--out", out, "--storage", "plain", "--positions")
    assert "сховище:      plain, позиції: так" in result.stdout
    result = run("index", corpus, "--out", out, "--limit", "1", "--no-memory")
    assert "документів:   1" in result.stdout
    assert "не мірялась" in result.stdout


def test_index_progress_goes_to_stderr_not_stdout(corpus: Path, tmp_path: Path):
    result = run("index", corpus, "--out", tmp_path / "i.bin", "--no-memory")
    assert result.exit_code == 0
    assert "індексація" not in result.stdout


def test_index_missing_corpus_dir(tmp_path: Path):
    result = run("index", tmp_path / "nope", "--out", tmp_path / "i.bin")
    assert result.exit_code == 1
    assert result.stderr.startswith("помилка:")
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_index_empty_dir(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    result = run("index", tmp_path / "empty", "--out", tmp_path / "i.bin")
    assert result.exit_code == 1
    assert ".txt" in result.stderr


def test_index_cannot_write_output(corpus: Path, tmp_path: Path):
    result = run("index", corpus, "--out", tmp_path / "no_such_dir" / "i.bin", "--no-memory")
    assert result.exit_code == 1
    assert result.stderr.startswith("помилка:")
    assert "Traceback" not in result.stderr


def test_index_rejects_bad_option_values(corpus: Path, tmp_path: Path):
    out = tmp_path / "i.bin"
    assert run("index", corpus, "--out", out, "--storage", "nope").exit_code == 2
    assert run("index", corpus, "--out", out, "--format", "yaml").exit_code == 2
    assert run("index", corpus, "--out", out, "--limit", "0").exit_code == 2


# --- search ---


def test_search_table_has_scores_and_snippets(built: Path):
    result = run("search", built, "whale")
    assert result.exit_code == 0, result.output
    line = next(line for line in result.stdout.splitlines() if "Whale Book" in line)
    assert "a.txt" in line
    assert "the white whale swims" in line
    assert "**" not in result.stdout  # підсвітка це стиль rich, а не зірочки


def test_search_no_snippets(built: Path):
    result = run("search", built, "whale", "--no-snippets")
    assert result.exit_code == 0
    assert "Whale Book" in result.stdout
    assert "swims" not in result.stdout


def test_search_json_lines(built: Path):
    result = run("search", built, "sea", "--json")
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["rank"] for r in rows] == [1, 2]
    assert {r["title"] for r in rows} == {"Whale Book", "Ship Book"}
    assert all(set(r) == {"rank", "doc_id", "score", "title", "path", "snippet"} for r in rows)
    assert all("**sea**" in r["snippet"] for r in rows)
    assert rows[0]["score"] >= rows[1]["score"]


def test_search_json_has_only_json_in_stdout(built: Path):
    result = run("search", built, "sea", "--json", "-vv")
    assert result.exit_code == 0
    for line in result.stdout.splitlines():
        json.loads(line)  # жодного сторонього рядка
    assert "DEBUG" in result.stderr  # логи окремо, у stderr


def test_search_json_no_snippets_gives_null(built: Path):
    result = run("search", built, "sea", "--json", "--no-snippets")
    assert all(json.loads(line)["snippet"] is None for line in result.stdout.splitlines())


def test_search_json_nothing_found_is_empty_output(built: Path):
    result = run("search", built, "zzzz", "--json")
    assert result.exit_code == 0
    assert result.stdout == ""


def test_search_nothing_found_message(built: Path):
    result = run("search", built, "zzzz")
    assert result.exit_code == 0
    assert "нічого не знайдено" in result.stdout


def test_phrase_and_parentheses(built: Path):
    # "the sea" є в обох книгах, тож дужки з ship/whale вирішують лише з NOT
    result = run("search", built, '"the sea" AND (ship OR whale)', "--json")
    assert len(result.stdout.splitlines()) == 2

    result = run("search", built, '"the sea" AND (ship OR whale) NOT whale', "--json")
    titles = [json.loads(line)["title"] for line in result.stdout.splitlines()]
    assert titles == ["Ship Book"]


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("engine", ["merge", "set"])
def test_scorer_and_engine_options(built: Path, scorer: str, engine: str):
    result = run("search", built, "sea", "--scorer", scorer, "--engine", engine, "--json")
    assert result.exit_code == 0, result.output
    assert len(result.stdout.splitlines()) == 2


def test_search_rejects_bad_choices(built: Path):
    assert run("search", built, "sea", "--scorer", "magic").exit_code == 2
    assert run("search", built, "sea", "--engine", "magic").exit_code == 2
    assert run("search", built, "sea", "--k", "0").exit_code == 2


def test_k_limits_shown_results(built: Path):
    result = run("search", built, "sea", "-k", "1", "--json")
    assert len(result.stdout.splitlines()) == 1
    result = run("search", built, "sea", "--k", "1", "--no-snippets")
    assert "знайдено документів: 2, показано: 1" in result.stdout


def test_verbose_shows_timing_and_cache_hit_in_stderr(built: Path):
    result = run("search", built, "sea", "--repeat", "2", "-vv")
    assert result.exit_code == 0
    assert "Searcher.search:" in result.stderr
    assert "cache hit: 'sea'" in result.stderr
    assert "load:" in result.stderr


def test_single_v_is_info_only(built: Path):
    result = run("search", built, "sea", "-v")
    assert "знайдено документів: 2" in result.stderr
    assert "DEBUG" not in result.stderr


def test_without_verbose_stderr_is_quiet(built: Path):
    result = run("search", built, "sea", "--repeat", "2")
    assert result.exit_code == 0
    assert result.stderr == ""


@pytest.mark.parametrize("query", ["(sea", "sea)", '"sea', "OR", ""])
def test_bad_query_is_one_line_error(built: Path, query: str):
    result = run("search", built, query)
    assert result.exit_code == 1
    assert result.stderr.startswith("помилка:")
    assert result.stderr.count("\n") == 1
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_phrase_without_positions_is_a_clean_error(corpus: Path, tmp_path: Path):
    out = tmp_path / "plain.bin"
    run("index", corpus, "--out", out, "--no-memory")
    result = run("search", out, '"the sea"')
    assert result.exit_code == 1
    assert "--positions" in result.stderr
    assert "Traceback" not in result.stderr


def test_search_in_missing_or_broken_file(tmp_path: Path):
    result = run("search", tmp_path / "nope.bin", "x")
    assert result.exit_code == 1
    assert result.stderr.startswith("помилка:")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"junk")
    result = run("search", junk, "x")
    assert result.exit_code == 1
    assert "Traceback" not in result.stderr


def test_error_keeps_going_after_exit_code_check(built: Path):
    # помилка не залишає за собою зіпсованого стану: наступна команда працює
    assert run("search", built, "(").exit_code == 1
    assert run("search", built, "sea", "--json").exit_code == 0


# --- stats ---


def test_stats_of_corpus(corpus: Path):
    result = run("stats", corpus, "--no-memory", "--top", "3")
    assert result.exit_code == 0, result.output
    assert "документів:   2" in result.stdout
    assert "lazy" in result.stdout
    assert "the" in result.stdout  # найчастіший терм у таблиці


def test_stats_of_corpus_measures_memory_by_default(corpus: Path):
    result = run("stats", corpus)
    assert result.exit_code == 0
    assert "tracemalloc" in result.stdout
    assert "МБ" in result.stdout


def test_stats_of_corpus_eager_matches_lazy(corpus: Path):
    lazy = run("stats", corpus, "--no-memory")
    eager = run("stats", corpus, "--no-memory", "--eager")
    assert "eager" in eager.stdout

    def counts(out: str) -> list[str]:
        return [line for line in out.splitlines() if line.startswith(("токенів", "словник"))]

    assert counts(lazy.stdout) == counts(eager.stdout) != []


def test_stats_of_index_file(built: Path):
    result = run("stats", built, "--top", "5")
    assert result.exit_code == 0, result.output
    assert "документів:   2" in result.stdout
    assert "термів:" in result.stdout
    assert "позиції: так" in result.stdout
    assert "sea" in result.stdout  # є в обох документах, тому в топі за df


def test_stats_of_missing_path(tmp_path: Path):
    result = run("stats", tmp_path / "nope")
    assert result.exit_code == 1
    assert result.stderr.startswith("помилка:")
    assert "Traceback" not in result.stderr


def test_stats_of_broken_index_file(tmp_path: Path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"junk")
    result = run("stats", junk)
    assert result.exit_code == 1
    assert "Traceback" not in result.stderr
