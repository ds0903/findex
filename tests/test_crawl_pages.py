from findex.crawl import parse_html

PAGE = """<!DOCTYPE html>
<html><head>
<title>  Мій   заголовок </title>
<style>p { color: red }</style>
<script>var secret = "no";</script>
</head><body>
<h1>Привіт</h1>
<p>Перший абзац, <b>жирний</b> текст &amp; ще.</p>
<p>Другий.<br>Рядок.</p>
<a href="/next">далі</a> <a href="https://other.test/x#top">чужий</a> <a>без href</a>
<noscript>вимкніть js</noscript>
</body></html>"""


def test_title_text_and_links():
    page = parse_html(PAGE)
    assert page.title == "Мій заголовок"
    assert page.links == ("/next", "https://other.test/x#top")
    assert "Привіт" in page.text
    assert "Перший абзац, жирний текст & ще." in page.text
    assert not page.noindex and not page.nofollow


def test_scripts_styles_and_noscript_are_not_text():
    text = parse_html(PAGE).text
    for junk in ("secret", "color", "вимкніть", "Мій заголовок"):
        assert junk not in text


def test_paragraphs_stay_on_separate_lines():
    lines = parse_html(PAGE).text.splitlines()
    assert "Привіт" in lines
    assert "Другий." in lines and "Рядок." in lines


def test_meta_robots():
    html = '<html><head><meta name="ROBOTS" content="NoIndex, follow"></head><body>x</body></html>'
    page = parse_html(html)
    assert page.noindex and not page.nofollow
    page = parse_html('<meta name="robots" content="nofollow"><a href="/a">a</a>')
    assert page.nofollow and not page.noindex
    assert parse_html('<meta name="robots" content="none">').noindex


def test_broken_html_does_not_crash():
    page = parse_html("<html><body><p>не закрито <a href='/x'>посилання <div></span>")
    assert page.links == ("/x",)
    assert "не закрито" in page.text


def test_empty_input():
    page = parse_html("")
    assert page.title == "" and page.text == "" and page.links == ()
