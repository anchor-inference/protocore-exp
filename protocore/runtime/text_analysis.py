"""Tokenising text for tool retrieval: identifiers, case, stopwords, stems.

One analyser serves both sides of a match. The tool catalogue and the query go
through the same steps, so a word reaches the index in exactly the form a query
will look it up in; a step applied to one side only is a match that silently
never happens.

The steps, in order:

1. **Identifier splitting.** Tool names and parameter names are identifiers,
   and a query says ``browser open`` where the catalogue says ``BrowserOpen``.
   Every identifier-shaped chunk yields its parts (CamelCase, ``snake_case``,
   ``kebab-case``, dotted and slashed paths, letter/digit boundaries) *and* the
   joined form, so both ``browser open`` and ``browseropen`` find it.
2. **Case folding** plus ``ё`` -> ``е``: Russian text is written both ways and
   the two spellings are the same word.
3. **Stopwords**, English and Russian, including the conversational fillers an
   operator types into a chat ("слушай", "короче", "плз", "please"). They carry
   no intent, and left in they match every description that happens to contain
   them.
4. **Stemming** with :mod:`protocore.runtime.stemmers`.
"""
# ruff: noqa: RUF001, RUF002, RUF003 — Cyrillic stopwords and examples are intentional

from __future__ import annotations

import re
from typing import Final

from protocore.runtime.stemmers import stem

_STOPWORDS_EN: Final[frozenset[str]] = frozenset(
    """
    a an the and or but of for in on at to from with without as is are was were
    be been being this that these those it its by if then else do does did have
    has had i you he she we they them us our your my me his her their not no so
    than too very can will just only also into any all each which what who when
    where how why there here please pls plz can could would should may might must
    shall let lets want need get got make im hey hi ok okay thanks thank
    """.split()
)

_STOPWORDS_RU: Final[frozenset[str]] = frozenset(
    """
    и в во не на что я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о об из ему теперь когда даже ну вдруг
    ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
    себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
    это этот эта эти этого этой мой моя мои мое твой наш наша наши ваш свой свою
    свое который которая которые которое какой какая какие очень просто тоже
    также сейчас пока тогда вообще чтобы чтоб
    пожалуйста плз плиз пжл пжлст слушай короче давай давайте можешь можете
    можно нужно нужен нужна хочу хотел хотела типа кстати ладно окей ок
    спасибо спс блин
    """.split()
)

#: Every stopword, already case-folded and with ``ё`` folded to ``е``.
STOPWORDS: Final[frozenset[str]] = _STOPWORDS_EN | _STOPWORDS_RU

_WORD: Final[re.Pattern[str]] = re.compile(r"\w+")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[\w\-./]+")
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[_\-./]+")
# lower -> Upper ("browserOpen"), Upper -> Upper+lower ("HTTPServer" splits as
# "HTTP" + "Server"), and letter <-> digit ("utf8", "e5small").
_CAMEL_LOWER_UPPER: Final[re.Pattern[str]] = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")
_CAMEL_ACRONYM: Final[re.Pattern[str]] = re.compile(r"(?<=[A-ZА-ЯЁ])(?=[A-ZА-ЯЁ][a-zа-яё])")
_LETTER_DIGIT: Final[re.Pattern[str]] = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def fold(text: str) -> str:
    """Case-fold ``text`` and spell ``ё`` as ``е``."""
    return text.casefold().replace("ё", "е")


def split_identifier(chunk: str) -> list[str]:
    """The folded parts of one identifier-shaped chunk.

    ``Mcp_Github_list_pull_requests`` -> ``mcp github list pull requests``;
    ``HTTPServer`` -> ``http server``; ``e5-small`` -> ``e 5 small``.
    """
    parts: list[str] = []
    for piece in _SEPARATORS.split(chunk):
        piece = _CAMEL_LOWER_UPPER.sub(" ", piece)
        piece = _CAMEL_ACRONYM.sub(" ", piece)
        piece = _LETTER_DIGIT.sub(" ", piece)
        parts.extend(fold(word) for word in piece.split())
    return parts


def raw_tokens(text: str) -> list[str]:
    """Folded words of ``text``, identifiers split with their joined form kept.

    Stopwords are still present; :func:`analyze` removes them.
    """
    tokens: list[str] = []
    for chunk in _IDENTIFIER.findall(text):
        parts = split_identifier(chunk)
        joined = fold(_SEPARATORS.sub("", chunk))
        if len(parts) > 1:
            tokens.extend(parts)
            tokens.append(joined)
        else:
            # A plain word, or a chunk whose separators left one part: the
            # word pattern drops whatever non-word character remains.
            tokens.extend(_WORD.findall(joined))
    return tokens


def content_words(text: str) -> list[str]:
    """Folded, identifier-split words of ``text`` with stopwords removed."""
    return [token for token in raw_tokens(text) if token not in STOPWORDS]


def analyze(text: str) -> list[str]:
    """Index terms of ``text``: :func:`content_words`, each stemmed."""
    return [stem(token) for token in content_words(text)]


__all__ = ["STOPWORDS", "analyze", "content_words", "fold", "raw_tokens", "split_identifier"]
