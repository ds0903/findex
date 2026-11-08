import pytest

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, follow, parse
from helpers import make_docs


def t(*names):
    return [Term(n) for n in names]


a, b, c, d = t("a", "b", "c", "d")


# --- тести на дерева, а не на рядки ---


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", a),
        ("a b", And(a, b)),
        ("a AND b", And(a, b)),
        ("a b c", And(And(a, b), c)),  # ліва асоціативність
        ("a OR b", Or(a, b)),
        ("a OR b c", Or(a, And(b, c))),  # AND сильніший за OR
        ("a b OR c", Or(And(a, b), c)),
        ("(a OR b) c", And(Or(a, b), c)),
        ("a OR b OR c", Or(Or(a, b), c)),
        ("NOT a", Not(a)),
        ("NOT a b", And(Not(a), b)),  # NOT чіпляється лише до найближчого операнда
        ("a NOT b", And(a, Not(b))),
        ("NOT (a OR b)", Not(Or(a, b))),
        ("NOT NOT a", Not(Not(a))),
        ("((a))", a),
        ('"a b"', Phrase(("a", "b"))),
        ('"a b c" d', And(Phrase(("a", "b", "c")), d)),
        ('"a"', a),  # фраза з одного слова - просто терм
        ("A OR B", Or(a, b)),  # терми нормалізуються
        ("a or b", And(And(a, Term("or")), b)),  # "or" малими літерами - звичайне слово
    ],
)
def test_parse_builds_expected_tree(query, tree):
    assert parse(query) == tree


def test_example_from_the_lab():
    tree = parse('python AND (async OR await) NOT java "event loop"')
    python, async_, await_, java = t("python", "async", "await", "java")
    expected = And(
        And(And(python, Or(async_, await_)), Not(java)),
        Phrase(("event", "loop")),
    )
    assert tree == expected


def test_a_or_b_c_is_or_of_and_not_and_of_or():
    assert parse("a OR b c") == Or(a, And(b, c))
    assert parse("a OR b c") != And(Or(a, b), c)


@pytest.mark.parametrize(
    "query",
    ["", "   ", "OR", "a OR", "AND a", "(a", "a)", "()", '"a b', '""', "NOT", "a (b"],
)
def test_bad_queries(query):
    with pytest.raises(QueryError):
        parse(query)


def test_error_messages_are_helpful():
    with pytest.raises(QueryError, match="закривн"):
        parse("(a OR b")
    with pytest.raises(QueryError, match="лапка"):
        parse('a "b')
    with pytest.raises(QueryError, match="порожній"):
        parse("")


def test_word_with_punctuation_becomes_terms():
    assert parse("foo.bar") == And(Term("foo"), Term("bar"))
    assert parse("Don’t") == Term("don't")  # той самий апостроф, що й у токенізаторі


def test_nodes_are_frozen_and_hashable():
    assert len({Term("a"), Term("a"), Term("b")}) == 2
    with pytest.raises(AttributeError):
        a.text = "x"  # type: ignore[misc]  # перевіряємо, що дерево незмінне
    assert hash(parse("a OR b c")) == hash(Or(a, And(b, c)))


# --- оператори Python на вузлах ---


def test_operators_build_nodes():
    assert a & b == And(a, b)
    assert a | b == Or(a, b)
    assert ~a == Not(a)
    assert (a | b) & ~c == And(Or(a, b), Not(c))


def test_operators_with_foreign_types():
    with pytest.raises(TypeError):
        a & 5  # type: ignore[operator]  # NotImplemented -> TypeError
    assert a != "a"


# --- str(дерево) знову розбирається в те саме дерево ---


@pytest.mark.parametrize(
    "tree",
    [
        a,
        And(a, And(b, c)),
        And(And(a, b), c),
        Or(a, Or(b, c)),
        Or(And(a, b), c),
        And(Or(a, b), c),
        Not(And(a, b)),
        Not(Not(a)),
        And(a, Not(Or(b, c))),
        Phrase(("x", "y")),
        Or(Phrase(("x", "y")), Not(d)),
    ],
)
def test_str_round_trips(tree):
    assert parse(str(tree)) == tree


# --- обчислення на індексі ---

# doc 0: the cat sat on the mat
# doc 1: the dog sat
# doc 2: a cat and a dog


@pytest.fixture
def index():
    return build_index(make_docs())


def test_evaluate_returns_sets(index):
    assert Term("cat").evaluate(index) == {0, 2}
    assert And(Term("cat"), Term("dog")).evaluate(index) == {2}
    assert Or(Term("cat"), Term("dog")).evaluate(index) == {0, 1, 2}
    assert Not(Term("cat")).evaluate(index) == {1}
    assert Term("zzz").evaluate(index) == set()


def test_terms_exclude_negated_and_have_no_duplicates():
    assert parse("a b NOT c a").terms() == ["a", "b"]
    assert parse('"x y" OR x').terms() == ["x", "y"]
    assert parse("NOT a").terms() == []


# --- фрази ---


def test_follow_two_pointer():
    assert follow([0, 4], [1, 5, 9]) == [1, 5]
    assert follow([0, 4], [2, 3]) == []
    assert follow([], [1]) == []
    assert follow([3], []) == []
    assert follow([1, 2, 3], [2, 3, 4]) == [2, 3, 4]


@pytest.fixture
def positional():
    return build_index(
        make_docs(
            [
                "the event loop runs",  # 0
                "loop of the event",  # 1: ті самі слова, але не підряд і не в тому порядку
                "an event, then a loop",  # 2: між словами є інші
                "run the event loop event loop",  # 3: фраза двічі
            ]
        ),
        positions=True,
    )


def test_phrase_requires_adjacent_words_in_order(positional):
    assert parse('"event loop"').evaluate(positional) == {0, 3}
    assert parse('"loop event"').evaluate(positional) == {3}  # у doc 1 слова є, але не підряд
    assert parse("event loop").evaluate(positional) == {0, 1, 2, 3}  # без лапок - просто AND


def test_longer_phrase_and_combinations(positional):
    assert parse('"the event loop"').evaluate(positional) == {0, 3}
    assert parse('"event loop" NOT run').evaluate(positional) == {0}
    assert parse('"event loop" OR "loop of"').evaluate(positional) == {0, 1, 3}
    assert parse('"event event"').evaluate(positional) == set()


def test_phrase_of_unknown_word_is_empty(positional):
    assert parse('"event zzz"').evaluate(positional) == set()


def test_phrase_without_positions_is_an_error(index):
    with pytest.raises(QueryError, match="--positions"):
        parse('"the cat"').evaluate(index)
