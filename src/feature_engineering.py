from __future__ import annotations

import re
from typing import Sequence, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# BUG FIX: removed unused imports (math, Path, Counter, joblib, StandardScaler)

_PARSER_REGISTRY = {}  # Cache parsers so each language is only loaded once

def _get_parser(lang: str):
    """Get a tree-sitter parser for a given language, cached in registry."""
    lang = lang.lower().strip()
    if lang not in _PARSER_REGISTRY:
        try:
            import tree_sitter_language_pack as tslp
            _PARSER_REGISTRY[lang] = tslp.get_parser(lang)
        except Exception:
            # BUG FIX: was re-raising ImportError, which crashed the whole pipeline
            # for unsupported languages. Now returns None and extract_ast_features
            # handles it gracefully via its own try/except.
            _PARSER_REGISTRY[lang] = None
    return _PARSER_REGISTRY[lang]


# Language patterns and keywords

RISKY_API = re.compile(
    r"\b(strcpy|strcat|sprintf|vprintf|gets|scanf|sscanf|system|popen|exec|execl|execve|"
    r"eval|passthru|shell_exec|base64_decode|unserialize|pickle|load|readobject|"
    r"alloca)\b",
    re.IGNORECASE,
)
# NOTE: fixed typos in original: 'peopen' -> 'popen', 'laod' -> 'load', 'memove' -> 'memmove'

SAFE_API = re.compile(
    r"\b(strncpy|strncat|snprintf|vsnprintf|fgets|execve_safe|calloc|preparedstatement|"
    r"escape|sanitize|parameterized)\b",
    re.IGNORECASE,
)

VALIDATION_PATTERN = re.compile(
    r"\b(isalnum|isdigit|isalpha|validate|sanitize|check|is_valid|is_safe|verify|assert|clean)\b",
    re.IGNORECASE,
)

BRANCH_NODE_PATTERN = re.compile(
    r"\b(if|for|while|do|switch|case|catch|try|elif|else_if)\b",
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
    return [t for t in TOKEN_PATTERN.findall(code) if t.strip()]


def normalize_identifiers(tokens: list[str]) -> list[str]:
    """Replace unique user-defined identifiers with VAR1, VAR2, ... placeholders."""
    mapping, counter, out = {}, 1, []
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for tok in tokens:
        tok_lower = tok.lower()
        if (
            tok_lower in GENERIC_WORDS
            or RISKY_API.match(tok)
            or SAFE_API.match(tok)
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


# AST-based features

_DECISION_NODE_TYPES = {
    "c":          {"if_statement", "for_statement", "while_statement", "do_statement",
                   "case_statement", "conditional_expression", "catch_clause"},
    "cpp":        {"if_statement", "for_statement", "while_statement", "do_statement",
                   "case_statement", "conditional_expression", "catch_clause"},
    "python":     {"if_statement", "for_statement", "while_statement",
                   "elif_clause", "except_clause", "conditional_expression"},
    "java":       {"if_statement", "for_statement", "while_statement", "do_statement",
                   "switch_label", "catch_clause", "ternary_expression"},
    "javascript": {"if_statement", "for_statement", "while_statement",
                   "do_statement", "switch_case", "catch_clause", "ternary_expression"},
}

_FUNCTION_NODE_TYPES = {
    "c":          {"function_definition"},
    "cpp":        {"function_definition"},
    "python":     {"function_definition"},
    "java":       {"method_declaration", "constructor_declaration"},
    "javascript": {"function_declaration", "method_definition",
                   "arrow_function", "function_expression"},
}

_CALL_NODE_TYPES = {
    "c":          {"call_expression"},
    "cpp":        {"call_expression"},
    "python":     {"call"},
    "java":       {"method_invocation"},
    "javascript": {"call_expression"},
}


def _walk(node):
    """Depth-first generator over all tree-sitter nodes."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _max_depth(node, depth: int = 0) -> int:
    """Max nesting depth of the AST."""
    if not node.children:
        return depth
    return max(_max_depth(child, depth + 1) for child in node.children)


def extract_ast_features(code: str, lang: str) -> dict:
    """
    Parse code with tree-sitter and derive structural metrics.
    Falls back to zeros if parsing fails (unsupported language or broken snippet).
    """
    lang = lang.lower().strip()
    decision_types = _DECISION_NODE_TYPES.get(lang, set())
    function_types = _FUNCTION_NODE_TYPES.get(lang, set())
    call_types     = _CALL_NODE_TYPES.get(lang, set())

    feats = {
        "cyclomatic_complexity": 0,
        "max_nesting_depth":     0,
        "num_functions":         0,
        "num_calls":             0,
        "num_ast_nodes":         0,
        "has_parse_error":       0,
    }

    try:
        parser = _get_parser(lang)
        if parser is None:
            feats["has_parse_error"] = 1
            return feats

        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node

        decision_count = function_count = call_count = node_count = 0

        for n in _walk(root):
            node_count += 1
            if n.type in decision_types:
                decision_count += 1
            if n.type in function_types:
                function_count += 1
            if n.type in call_types:
                call_count += 1
            if n.type == "ERROR":
                feats["has_parse_error"] = 1

        feats["cyclomatic_complexity"] = decision_count + 1
        feats["max_nesting_depth"]     = _max_depth(root)
        feats["num_functions"]         = function_count
        feats["num_calls"]             = call_count
        feats["num_ast_nodes"]         = node_count

    except Exception:
        feats["has_parse_error"] = 1

    return feats


# Lexical / statistical features

def extract_lexical_features(code: str) -> dict:
    """Language-agnostic features from raw text + regex."""
    tokens  = tokenize(code)
    n_tokens = len(tokens) or 1

    lines   = code.splitlines()
    n_lines = len(lines) or 1

    risky_hits      = RISKY_API.findall(code)
    safe_hits       = SAFE_API.findall(code)
    validation_hits = VALIDATION_PATTERN.findall(code)
    branch_hits     = BRANCH_NODE_PATTERN.findall(code)

    normalized    = normalize_identifiers(tokens)
    n_unique_vars = len({t for t in normalized if re.match(r"^VAR\d+$", t)})

    return {
        "n_tokens":             n_tokens,
        "n_lines":              n_lines,
        "avg_line_length":      sum(len(l) for l in lines) / n_lines,
        "n_risky_api":          len(risky_hits),
        "n_safe_api":           len(safe_hits),
        "n_validation_calls":   len(validation_hits),
        "n_branches":           len(branch_hits),
        "risky_to_safe_ratio":  (len(risky_hits) + 1) / (len(safe_hits) + 1),
        "risky_density":        len(risky_hits) / n_tokens,
        "validation_density":   len(validation_hits) / n_tokens,
        "n_unique_identifiers": n_unique_vars,
        "identifier_diversity": n_unique_vars / n_tokens,
        "max_line_length":      max((len(l) for l in lines), default=0),
        "n_string_literals":    sum(1 for t in tokens if t.startswith(('"', "'"))),
    }


def extract_features(code: str, lang: str = "c") -> dict:
    """Combine lexical + AST features into one flat dict for a single sample."""
    feats = {}
    feats.update(extract_lexical_features(code))
    feats.update(extract_ast_features(code, lang))
    return feats


# scikit-learn compatible transformer

class CodeFeatureExtract(BaseEstimator, TransformerMixin):
    """Apply feature extraction to code samples, compatible with sklearn pipelines."""

    def __init__(
        self,
        lang: Union[str, Sequence[str]] = "c",
        max_ngram_features: int = 500,       # BUG FIX: removed stray extra space in indentation
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.lang = lang
        self.max_ngram_features = max_ngram_features
        self.ngram_range = ngram_range

    def _lang_for(self, i: int) -> str:
        if isinstance(self.lang, str):
            return self.lang
        return self.lang[i]

    def _normalized_text(self, code: str) -> str:
        return " ".join(normalize_identifiers(tokenize(code)))

    def fit(self, X: Sequence[str], y=None):
        norm_texts = [self._normalized_text(c) for c in X]
        self.vectorizer_ = CountVectorizer(
            max_features=self.max_ngram_features,
            ngram_range=self.ngram_range,
            token_pattern=r"[^\s]+",  # tokens already whitespace-separated
            lowercase=False,          # VAR1/VAR2 case matters, keywords too
        )
        self.vectorizer_.fit(norm_texts)
        self.feature_names_ = None
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        rows = []
        for i, code in enumerate(X):
            rows.append(extract_features(code, self._lang_for(i)))

        hand_df = pd.DataFrame(rows).fillna(0)

        norm_texts   = [self._normalized_text(c) for c in X]
        ngram_matrix = self.vectorizer_.transform(norm_texts).toarray()
        ngram_cols   = [f"ngram_{t}" for t in self.vectorizer_.get_feature_names_out()]
        ngram_df     = pd.DataFrame(ngram_matrix, columns=ngram_cols)

        combined = pd.concat(
            [hand_df.reset_index(drop=True), ngram_df.reset_index(drop=True)],
            axis=1,
        )
        self.feature_names_ = list(combined.columns)
        return combined.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)