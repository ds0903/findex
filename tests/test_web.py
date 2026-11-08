import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.documents import iter_documents
from findex.index import Index, build_index
from findex.searcher import Searcher
from findex.store import save
from findex.web import create_app
from findex.web.deps import get_index, get_searcher
from findex.web.service import highlight_spans, mark
from findex.web.settings import Settings


@pytest.fixture
def index(corpus: Path) -> Index:
    return build_index(iter_documents(corpus), positions=True, root=str(corpus))


@pytest.fixture
def app(index: Index):
    """Застосунок без lifespan і без файлу індексу: індекс підсувається через overrides."""
    app = create_app(Settings(index_path=Path("нема-такого-файлу.bin")))
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_searcher] = lambda: Searcher(index)
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def page_client(app, index: Index) -> TestClient:
    """Сторінки читають стан застосунку напряму, тож кладемо індекс і туди."""
    app.state.findex.index = index
    app.state.findex.searcher = Searcher(index)
    return TestClient(app)


# --- /search ---


def test_search_returns_ranked_results_with_snippets(client: TestClient):
    response = client.get("/search", params={"q": "whale"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "whale" and body["total"] == 1 and body["pages"] == 1
    assert body["scorer"] == "bm25" and body["took_ms"] >= 0
    (hit,) = body["results"]
    assert hit["title"] == "Whale Book" and hit["score"] > 0
    assert "whale" in hit["snippet"]
    start, end = hit["highlights"][0]
    assert hit["snippet"][start:end].lower() == "whale"


def test_search_pagination(client: TestClient):
    first = client.get("/search", params={"q": "sea", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "sea", "k": 1, "page": 2}).json()
    beyond = client.get("/search", params={"q": "sea", "k": 1, "page": 3}).json()
    assert first["total"] == second["total"] == 2 and first["pages"] == 2
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    assert beyond["results"] == []


def test_search_scorers_give_scores(client: TestClient):
    for scorer in ("bm25", "tfidf"):
        body = client.get("/search", params={"q": "sea", "scorer": scorer}).json()
        assert body["scorer"] == scorer and len(body["results"]) == 2


def test_search_supports_the_query_language(client: TestClient):
    body = client.get("/search", params={"q": "sea AND NOT whale"}).json()
    assert [r["title"] for r in body["results"]] == ["Ship Book"]
    body = client.get("/search", params={"q": '"white whale"'}).json()
    assert [r["title"] for r in body["results"]] == ["Whale Book"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "x" * 201},
        {"q": "sea", "k": 0},
        {"q": "sea", "k": 51},
        {"q": "sea", "k": "many"},
        {"q": "sea", "page": 0},
        {"q": "sea", "page": 101},
        {"q": "sea", "scorer": "bm26"},
    ],
)
def test_bad_parameters_are_422_with_detail(client: TestClient, params: dict[str, object]):
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_bad_query_syntax_is_422(client: TestClient):
    response = client.get("/search", params={"q": "whale AND"})
    assert response.status_code == 422
    assert "некоректний запит" in response.json()["detail"]


# --- /docs/{doc_id} ---


def test_document(client: TestClient):
    body = client.get("/docs/0").json()
    assert body["doc_id"] == 0 and body["title"] == "Whale Book"
    assert "white whale" in body["text"] and body["truncated"] is False and body["tokens"] > 0


def test_document_text_is_cut_by_limit(client: TestClient):
    body = client.get("/docs/0", params={"limit": 5}).json()
    assert len(body["text"]) == 5 and body["truncated"] is True


def test_unknown_document_is_404(client: TestClient):
    response = client.get("/docs/99")
    assert response.status_code == 404
    assert "99" in response.json()["detail"]


def test_document_id_must_be_a_number(client: TestClient):
    assert client.get("/docs/whale").status_code == 422


# --- /stats, /health ---


def test_stats(client: TestClient, index: Index):
    body = client.get("/stats").json()
    assert body["documents"] == 2 and body["terms"] == len(index)
    assert body["uptime_seconds"] >= 0 and body["avg_doc_length"] > 0 and body["version"]


def test_health_ok(client: TestClient):
    assert client.get("/health").json() == {"status": "ok", "documents": 2}


def test_health_is_503_while_index_is_not_loaded():
    client = TestClient(create_app(Settings(index_path=Path("нема.bin"))))
    response = client.get("/health")
    assert response.status_code == 503
    assert "не завантажений" in response.json()["detail"]
    assert client.get("/search", params={"q": "sea"}).status_code == 503


# --- помилки й id запиту ---


def test_unexpected_error_is_logged_with_request_id_and_hidden_from_client(
    app, caplog: pytest.LogCaptureFixture
):
    def broken() -> Index:
        raise RuntimeError("секрет бази 12345")

    app.dependency_overrides[get_index] = broken
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        response = client.get("/stats", headers={"X-Request-ID": "req-42"})
    assert response.status_code == 500
    assert response.json() == {"detail": "внутрішня помилка сервера", "request_id": "req-42"}
    assert "секрет" not in response.text and "Traceback" not in response.text
    assert response.headers["x-request-id"] == "req-42"
    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "req-42" in record.getMessage()
    assert record.exc_info is not None and "секрет бази" in str(record.exc_info[1])


def test_request_id_is_generated_and_unsafe_ids_are_replaced(client: TestClient):
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 12
    replaced = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})
    assert replaced.headers["x-request-id"] != "bad id\twith spaces"


def test_openapi_and_swagger(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert client.get("/docs").status_code == 200  # Swagger UI не заслонений /docs/{doc_id}


# --- lifespan і Settings ---


def test_lifespan_loads_index_once(tmp_path: Path, index: Index, corpus: Path):
    path = tmp_path / "index.bin"
    save(index, path)
    with TestClient(create_app(Settings(index_path=path))) as client:
        assert client.get("/health").json()["documents"] == 2
        stats = client.get("/stats").json()
        assert stats["postings"] == index.num_postings() and stats["index_file_mb"] is not None
        assert client.get("/search", params={"q": "whale"}).json()["total"] == 1


def test_lifespan_without_index_file_keeps_service_up_but_unhealthy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        with TestClient(create_app(Settings(index_path=tmp_path / "нема.bin"))) as client:
            assert client.get("/health").status_code == 503
    assert any("не вдалося завантажити індекс" in r.getMessage() for r in caplog.records)


def test_corpus_root_overrides_the_path_stored_in_the_index(
    tmp_path: Path, index: Index, corpus: Path
):
    index.root = str(tmp_path / "був-тут-під-час-індексації")
    path = tmp_path / "index.bin"
    save(index, path)
    with TestClient(create_app(Settings(index_path=path, corpus_root=corpus))) as client:
        hit = client.get("/search", params={"q": "whale"}).json()["results"][0]
        assert "whale" in hit["snippet"]


def test_settings_come_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "x.bin"))
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.index_path == tmp_path / "x.bin" and settings.corpus_root == tmp_path
    assert settings.log_level == "debug"


# --- HTML ---


def test_home_page(page_client: TestClient):
    response = page_client.get("/")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert '<form action="/"' in response.text and "59 книгах" in response.text


def test_results_page_shows_scores_and_highlights(page_client: TestClient):
    html = page_client.get("/", params={"q": "whale"}).text
    assert "Whale Book" in html and "<mark>whale</mark>" in html.lower()
    assert 'class="score"' in html and "Знайдено документів: <strong>1</strong>" in html
    assert "/view/0?q=whale" in html


def test_results_page_has_pagination(page_client: TestClient):
    html = page_client.get("/", params={"q": "sea", "k": 5}).text
    assert 'class="pager"' not in html  # усе вмістилося на одній сторінці
    html = page_client.get("/", params={"q": "sea", "k": 1}).text
    assert 'class="pager"' in html and "Далі" in html and "page=2" in html
    html = page_client.get("/", params={"q": "sea", "k": 1, "page": 2}).text
    assert "Назад" in html and 'aria-current="page">2<' in html


def test_page_shows_query_error_not_a_stack_trace(page_client: TestClient):
    response = page_client.get("/", params={"q": "whale AND"})
    assert response.status_code == 200 and "Некоректний запит" in response.text


def test_query_is_escaped_in_html(page_client: TestClient):
    html = page_client.get("/", params={"q": '<script>alert("x")</script>'}).text
    assert "<script>alert" not in html


def test_page_clamps_wild_parameters(page_client: TestClient):
    response = page_client.get("/", params={"q": "sea", "k": 9999, "page": -4, "scorer": "nope"})
    assert response.status_code == 200 and "BM25" in response.text


def test_page_before_index_is_loaded(app):
    html = TestClient(app).get("/", params={"q": "sea"}).text
    assert "ще завантажується" in html


def test_document_view(page_client: TestClient):
    html = page_client.get("/view/0", params={"q": "whale"}).text
    assert "Whale Book" in html and "<mark>whale</mark>" in html.lower()
    assert page_client.get("/view/99").status_code == 404


def test_static_css(client: TestClient):
    response = client.get("/static/style.css")
    assert response.status_code == 200 and "text/css" in response.headers["content-type"]


# --- службові функції ---


def test_mark_escapes_html():
    out = mark("<b>whale</b> & co", [(3, 8)])
    assert str(out) == "&lt;b&gt;<mark>whale</mark>&lt;/b&gt; &amp; co"


def test_highlight_spans_follow_the_tokenizer():
    text = "The White-Whale's whales whale."
    spans = highlight_spans(text, {"whale"})
    assert [text[a:b] for a, b in spans] == ["whale"]  # інші форми не є термом "whale"
