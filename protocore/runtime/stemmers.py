"""Russian and English stemmers for tool retrieval, in pure Python.

Both are fresh implementations of published algorithms, written for this
package; no code is taken from an existing stemming library.

* Russian follows the Snowball Russian stemmer (Martin Porter's algorithm as
  published on the Snowball project site). The steps are written as regular
  expressions over the RV region, which keeps the whole algorithm to one page.
* English follows the original Porter stemmer (M. F. Porter, "An algorithm for
  suffix stripping", 1980), not the later Porter2 revision.

Retrieval needs only that a word and its inflections reduce to the same key on
both sides of the match, so the one departure from the published algorithms is
deliberate: a token shorter than :data:`MIN_STEM_LENGTH` is returned unchanged.
Two-letter tokens are abbreviations and identifiers far more often than words
("id", "ui", "db"), and stripping a letter from them merges unrelated terms.
"""
# ruff: noqa: RUF001, RUF002, RUF003 — Cyrillic suffix tables and examples are intentional

from __future__ import annotations

import re
from typing import Final

#: Tokens shorter than this are never stemmed (see the module docstring).
MIN_STEM_LENGTH: Final[int] = 3

_CYRILLIC: Final[re.Pattern[str]] = re.compile(r"[а-яё]")

# Russian: every pattern is anchored at the end and applied to the RV region
# only. A suffix that the algorithm removes "when preceded by а or я" is
# written with a lookbehind, and because the lookbehind runs over RV as well,
# the а/я has to lie inside RV exactly as the algorithm requires.
_RU_VOWELS: Final[str] = "аеиоуыэюя"
_RU_RV: Final[re.Pattern[str]] = re.compile(r"^(.*?[аеиоуыэюя])(.*)$")
_RU_PERFECTIVE_GERUND: Final[re.Pattern[str]] = re.compile(
    r"((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$"
)
_RU_REFLEXIVE: Final[re.Pattern[str]] = re.compile(r"(с[яь])$")
_RU_ADJECTIVE: Final[re.Pattern[str]] = re.compile(
    r"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$"
)
_RU_PARTICIPLE: Final[re.Pattern[str]] = re.compile(
    r"((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$"
)
_RU_VERB: Final[re.Pattern[str]] = re.compile(
    r"((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)"
    r"|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$"
)
_RU_NOUN: Final[re.Pattern[str]] = re.compile(
    r"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$"
)
_RU_FINAL_I: Final[re.Pattern[str]] = re.compile(r"и$")
_RU_DERIVATIONAL: Final[re.Pattern[str]] = re.compile(r"ость?$")
_RU_SUPERLATIVE: Final[re.Pattern[str]] = re.compile(r"(ейше|ейш)$")
_RU_DOUBLE_N: Final[re.Pattern[str]] = re.compile(r"нн$")
_RU_SOFT_SIGN: Final[re.Pattern[str]] = re.compile(r"ь$")


def _ru_r2_start(word: str) -> int:
    """Start of the Snowball R2 region of ``word``.

    R1 begins after the first consonant that follows a vowel; R2 is the same
    rule applied again inside R1.
    """

    def region_after(start: int) -> int:
        for index in range(start + 1, len(word)):
            if word[index] not in _RU_VOWELS and word[index - 1] in _RU_VOWELS:
                return index + 1
        return len(word)

    return region_after(region_after(0))


def stem_russian(word: str) -> str:
    """Snowball Russian stem of a lower-case word; ``ё`` is folded to ``е``."""
    word = word.replace("ё", "е")
    if len(word) < MIN_STEM_LENGTH:
        return word
    match = _RU_RV.match(word)
    if match is None:
        return word
    prefix, rv = match.groups()

    # Step 1: a perfective gerund, or else reflexive + adjectival / verb / noun.
    stripped = _RU_PERFECTIVE_GERUND.sub("", rv, count=1)
    if stripped != rv:
        rv = stripped
    else:
        rv = _RU_REFLEXIVE.sub("", rv, count=1)
        stripped = _RU_ADJECTIVE.sub("", rv, count=1)
        if stripped != rv:
            rv = _RU_PARTICIPLE.sub("", stripped, count=1)
        else:
            stripped = _RU_VERB.sub("", rv, count=1)
            rv = _RU_NOUN.sub("", rv, count=1) if stripped == rv else stripped

    # Step 2: a trailing и.
    rv = _RU_FINAL_I.sub("", rv, count=1)

    # Step 3: a derivational ость / ост, only when it lies inside R2.
    whole = prefix + rv
    derivational = _RU_DERIVATIONAL.search(whole)
    if derivational is not None and derivational.start() >= _ru_r2_start(whole):
        rv = rv[: len(rv) - (len(whole) - derivational.start())]

    # Step 4: a soft sign, or else a superlative and then нн -> н.
    stripped = _RU_SOFT_SIGN.sub("", rv, count=1)
    if stripped != rv:
        rv = stripped
    else:
        rv = _RU_SUPERLATIVE.sub("", rv, count=1)
        rv = _RU_DOUBLE_N.sub("н", rv, count=1)
    return prefix + rv


def _is_consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in "aeiou":
        return False
    if letter == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True


def _measure(word: str) -> int:
    """Porter's *m*: the number of vowel-consonant sequences in ``word``."""
    count, index, length = 0, 0, len(word)
    while index < length and _is_consonant(word, index):
        index += 1
    while index < length:
        while index < length and not _is_consonant(word, index):
            index += 1
        if index >= length:
            break
        while index < length and _is_consonant(word, index):
            index += 1
        count += 1
    return count


def _has_vowel(word: str) -> bool:
    return any(not _is_consonant(word, index) for index in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """Consonant-vowel-consonant ending whose last letter is not w, x or y."""
    return (
        len(word) >= 3
        and _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


# Suffix tables are ordered so the longest candidate is tried first wherever two
# of them share an ending ("ational" before "tional", "ement" before "ent").
# Step 2 is the table of the 1980 paper: "abli" -> "able", and no "logi" rule.
# Both of those were later changes in Porter's own code, and taking them would
# make this stemmer disagree with the published algorithm it claims to be.
_EN_STEP2: Final[tuple[tuple[str, str], ...]] = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)
_EN_STEP3: Final[tuple[tuple[str, str], ...]] = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_EN_STEP4: Final[tuple[str, ...]] = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def stem_english(word: str) -> str:
    """Porter (1980) stem of a lower-case ASCII word."""
    if len(word) < MIN_STEM_LENGTH:
        return word

    # Step 1a: plurals.
    if word.endswith("sses") or word.endswith("ies"):
        word = word[:-2]
    elif not word.endswith("ss") and word.endswith("s"):
        word = word[:-1]

    # Step 1b: past tense and progressive.
    tidy_up = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word, tidy_up = word[:-2], True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word, tidy_up = word[:-3], True
    if tidy_up:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # Step 1c: a terminal y after a vowel becomes i.
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Steps 2 and 3: map double suffixes to single ones. Only the first
    # matching suffix is considered, whether or not its condition holds.
    for table in (_EN_STEP2, _EN_STEP3):
        for suffix, replacement in table:
            if word.endswith(suffix):
                if _measure(word[: -len(suffix)]) > 0:
                    word = word[: -len(suffix)] + replacement
                break

    # Step 4: strip a suffix when what remains is long enough.
    for suffix in _EN_STEP4:
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _measure(base) > 1 and (suffix != "ion" or base.endswith(("s", "t"))):
                word = base
            break

    # Step 5: a final e, and a double l.
    if word.endswith("e"):
        base = word[:-1]
        measure = _measure(base)
        if measure > 1 or (measure == 1 and not _ends_cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


def stem(token: str) -> str:
    """Stem one folded token with the stemmer its script calls for.

    Tokens mixing digits, other scripts or symbols are returned as they are:
    there is no inflection in ``utf8`` or ``e5`` to remove.
    """
    if _CYRILLIC.search(token):
        return stem_russian(token)
    if token.isascii() and token.isalpha():
        return stem_english(token)
    return token


__all__ = ["MIN_STEM_LENGTH", "stem", "stem_english", "stem_russian"]
