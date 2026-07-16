from __future__ import annotations

import re 
import math
from pathlib import Path
from collections import Counter
from typing import Sequence, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

_PARSER_REGISTRY = {}                                                                   #create a dictionary to store parsers

def _get_parser(lang: str):
    """Get a dependency parser for a given language."""
    lang = lang.lower().strip()
    if lang not in _PARSER_REGISTRY:
        try:
            import tree_sitter_language_pack as tslp
            _PARSER_REGISTRY[lang] = tslp.get_parser(lang)
        except Exception:
            raise ImportError(f"Couldn't load parser for the language '{lang}'")
    return _PARSER_REGISTRY[lang]

#language patterns and keywords

RISKY_API = re.compile(
    r"\b(strcpy|strcat|sprintf|vprintf|gets|scanf|sscanf|system|peopen|exec|execl|execve|"
    r"eval|passthru|shell_exec|base64_decode|unserialize|pickle|laod|readobject|"
    r"memcpy|memove|malloc|realloc|free|alloca|printf|fprintf)\b",
    re.IGNORECASE,
)

SAFE_API = re.compile(
    r"\b(strncpy|strncat|snprintf|vsnprintf|fgets|execve_safe|calloc|preparedstatement|"
    r"escape|sanitize|parameterized)\b",
    re.IGNORECASE,
)

VALIDATION_PATTERN = re.compile(
    r"\b(isalnum|isdigit|isalpha|validate|sanitize|check|is_valid|is_safe|verify|assert|clean)\b",
    re.IGNORECASE
)

BRANCH_NODE_PATTERN = re.compile(
    r"(if|for|while|do|switch|case|catch|try|conditional|elif|else_if)",
    re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(
    r'\d+\.\d+|\d+|"[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_]*|'
    r'[{}()\[\];,]|==|!=|<=|>=|->|&&|\|\||.'
)

GENERIC_WORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "try", "catch", 
    "finally", "return", "class", "function", "def", "fn", "var", "let", "const",
    "int", "char", "void", "string", "bool", "boolean", "float", "double", "import", 
    "include", "package", "using", "public", "private", "protected", "static", "new",
    "self", "this", "struct", "enum", "union", "typedef", "sizeof", "lambda", "async",
    "await"
}

def tokenize(code: str) -> list[str]:
    return [t for t in TOKEN_PATTERN.findall(code) if t.strip()]           #filter empty tokens or white spaces and ignore them

def normalize_identifiers(tokens: list[str]) -> list[str]:
    mapping, counter, out = {}, 1, []                                     #if same username appears multiple times, it is mapped to same ID
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for tok in tokens:
        tok_lower=tok.lower()
        if (
            tok_lower in GENERIC_WORDS
            or RISKY_API.match(tok)
            or SAFE_API.match(tok)                                        #to find whether the above mentioned token are available in the source codes
            or VALIDATION_PATTERN.match(tok)
            or not ident.match(tok)
            or len(tok) <= 2
        ):
            out.append(tok)
            continue

        if tok not in mapping:
            mapping[tok] = f"VAR{counter}"
            counter += 1
        out.append(mapping[tok])
    return out




# -----------------------------------------------------------------------------    