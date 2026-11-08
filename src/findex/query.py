"""Мова запитів і її дерево.

Приклад: python AND (async OR await) NOT java "event loop"

Граматика (рекурсивний спуск), пріоритет: NOT вищий за AND, AND вищий за OR:

    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr ('AND'? not_expr)*      два терми поспіль - це AND
    not_expr -> 'NOT' not_expr | atom
    atom     -> слово | "фраза" | '(' or_expr ')'

Оператори пишуться великими літерами; "or" малими - звичайне слово. Слова приводяться до
термів тим самим токенізатором, що й текст документів.

Парсер повертає дерево з вузлів Term / Phrase / And / Or / Not. Вузли можна збирати
й руками: Term("a") & ~Term("b") == And(Term("a"), Not(Term("b"))).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from findex.index import Index
from findex.tokenizer import tokenize


class QueryError(ValueError):
    """Запит написано неправильно, або його не можна виконати на цьому індексі."""


class Node:
    """Вершина дерева запиту."""

    def evaluate(self, index: Index) -> set[int]:
        """Множина doc_id документів, які задовольняють цю вершину."""
        raise NotImplementedError

    def terms(self) -> list[str]:
        """Терми, які мають підвищувати рейтинг (все, що не під NOT), без повторів."""
        raise NotImplementedError

    def __and__(self, other: Node) -> And:
        # перевірка потрібна в runtime (`a & 5` має дати TypeError), хоч за анотацією other це Node
        if not isinstance(other, Node):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return And(self, other)

    def __or__(self, other: Node) -> Or:
        if not isinstance(other, Node):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids(self.text))

    def terms(self) -> list[str]:
        return [self.text]

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        return phrase_docs(index, self.words)

    def terms(self) -> list[str]:
        return list(dict.fromkeys(self.words))

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        found = self.left.evaluate(index)
        if not found:  # правий бік обчислювати нема сенсу
            return found
        return found & self.right.evaluate(index)

    def terms(self) -> list[str]:
        return list(dict.fromkeys(self.left.terms() + self.right.terms()))

    def __str__(self) -> str:
        left = f"({self.left})" if isinstance(self.left, Or) else str(self.left)
        right = f"({self.right})" if isinstance(self.right, And | Or) else str(self.right)
        return f"{left} AND {right}"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> list[str]:
        return list(dict.fromkeys(self.left.terms() + self.right.terms()))

    def __str__(self) -> str:
        right = f"({self.right})" if isinstance(self.right, Or) else str(self.right)
        return f"{self.left} OR {right}"


@dataclass(frozen=True)
class Not(Node):
    operand: Node

    def evaluate(self, index: Index) -> set[int]:
        return set(index.all_doc_ids()) - self.operand.evaluate(index)

    def terms(self) -> list[str]:
        return []

    def __str__(self) -> str:
        inner = f"({self.operand})" if isinstance(self.operand, And | Or) else str(self.operand)
        return f"NOT {inner}"


# --- фрази: двовказівниковий прохід по позиціях ---


def follow(prev: Sequence[int], nxt: Sequence[int]) -> list[int]:
    """Позиції з nxt, які стоять одразу після якоїсь позиції з prev (обидва списки відсортовані)."""
    out: list[int] = []
    i = j = 0
    while i < len(prev) and j < len(nxt):
        want = prev[i] + 1
        if nxt[j] == want:
            out.append(nxt[j])
            i += 1
            j += 1
        elif nxt[j] < want:
            j += 1
        else:
            i += 1
    return out


def phrase_docs(index: Index, words: Sequence[str]) -> set[int]:
    """Документи, де слова йдуть підряд. Потребує індексу з позиціями."""
    if not index.positions:
        raise QueryError("фразовий запит потребує позицій: побудуйте індекс з --positions")

    candidates = set(index.doc_ids(words[0]))
    for word in words[1:]:
        candidates &= set(index.doc_ids(word))
    if not candidates:
        return set()

    # для кожного слова: doc_id -> позиції (лише для документів-кандидатів)
    where = [
        {p.doc_id: p.positions for p in index.iter_postings(word) if p.doc_id in candidates}
        for word in words
    ]
    found: set[int] = set()
    for doc_id in candidates:
        ends = list(where[0][doc_id])
        for table in where[1:]:
            ends = follow(ends, table[doc_id])
            if not ends:
                break
        if ends:
            found.add(doc_id)
    return found


# --- лексер ---

_LEX = re.compile(r'\s*(?:(?P<lpar>\()|(?P<rpar>\))|"(?P<phrase>[^"]*)"|(?P<word>[^\s()"]+))')
_KEYWORDS = {"AND", "OR", "NOT"}


@dataclass(frozen=True)
class _Token:
    kind: str  # LPAR, RPAR, PHRASE, WORD, AND, OR, NOT
    text: str
    pos: int


def _lex(query: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while query[pos:].strip():
        match = _LEX.match(query, pos)
        if match is None:
            raise QueryError(f"незакрита лапка на позиції {query.find('"', pos)}")
        kind = match.lastgroup
        if kind == "lpar":
            tokens.append(_Token("LPAR", "(", match.start(kind)))
        elif kind == "rpar":
            tokens.append(_Token("RPAR", ")", match.start(kind)))
        elif kind == "phrase":
            tokens.append(_Token("PHRASE", match.group(kind), match.start(kind) - 1))
        elif kind == "word":
            word = match.group(kind)
            if word in _KEYWORDS:
                tokens.append(_Token(word, word, match.start(kind)))
            elif any(True for _ in tokenize(word)):
                tokens.append(_Token("WORD", word, match.start(kind)))
            # слово з самої пунктуації ("-", "...") пропускаємо
        pos = match.end()
    return tokens


# --- парсер ---


class _Parser:
    def __init__(self, tokens: Iterable[_Token]) -> None:
        self.tokens = list(tokens)
        self.i = 0

    def peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def parse(self) -> Node:
        if not self.tokens:
            raise QueryError("порожній запит")
        node = self.or_expr()
        extra = self.peek()
        if extra is not None:
            raise QueryError(f"зайвий {extra.text!r} на позиції {extra.pos}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while (token := self.peek()) is not None and token.kind == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while (token := self.peek()) is not None:
            if token.kind == "AND":
                self.take()
            elif token.kind not in {"WORD", "PHRASE", "LPAR", "NOT"}:
                break  # OR або ")" - це вже не наша справа
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        token = self.peek()
        if token is not None and token.kind == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        token = self.peek()
        if token is None:
            raise QueryError("запит обривається: очікувався терм")
        if token.kind == "LPAR":
            self.take()
            node = self.or_expr()
            closing = self.peek()
            if closing is None or closing.kind != "RPAR":
                raise QueryError(f"немає закривної дужки для дужки на позиції {token.pos}")
            self.take()
            return node
        if token.kind == "WORD":
            self.take()
            return _word_node(token.text)
        if token.kind == "PHRASE":
            self.take()
            return _phrase_node(token.text, token.pos)
        raise QueryError(f"очікувався терм, а знайдено {token.text!r} на позиції {token.pos}")


def _word_node(word: str) -> Node:
    # одне слово запиту може дати кілька термів ("foo.bar"), вони йдуть як AND
    terms = list(tokenize(word))
    node: Node = Term(terms[0])
    for term in terms[1:]:
        node = And(node, Term(term))
    return node


def _phrase_node(text: str, pos: int) -> Node:
    words = tuple(tokenize(text))
    if not words:
        raise QueryError(f"порожня фраза на позиції {pos}")
    if len(words) == 1:
        return Term(words[0])
    return Phrase(words)


def parse(query: str) -> Node:
    """Розбирає запит у дерево. Кидає QueryError з поясненням, якщо запит поганий."""
    return _Parser(_lex(query)).parse()
