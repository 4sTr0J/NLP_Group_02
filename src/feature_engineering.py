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
            or len(tok) <= 2 #check if token is 1 or 2 characters long
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
    "c": {"if_statement", "for_statement", "while_statement", "do_statement",
          "case_statement", "conditional_expression", "catch_clause"},
    "cpp": {"if_statement", "for_statement", "while_statement", "do_statement",
            "case_statement", "conditional_expression", "catch_clause"},
    "python": {"if_statement", "for_statement", "while_statement",
               "elif_clause", "except_clause", "conditional_expression"},
    "java": {"if_statement", "for_statement", "while_statement", "do_statement",
              "switch_label", "catch_clause", "ternary_expression"},
    "javascript": {"if_statement", "for_statement", "while_statement",
                   "do_statement", "switch_case", "catch_clause",
                   "ternary_expression"},
}

_FUNCTION_NODE_TYPES = {
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "python": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "javascript": {"function_declaration", "method_definition",
                   "arrow_function", "function_expression"},
}

_CALL_NODE_TYPES = {
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "python": {"call"},
    "java": {"method_invocation"},
    "javascript": {"call_expression"},
}


#we apply DFS to save memory
def _walk(node):
    """Depth-first generator over all tree-sitter nodes."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _max_depth(node, depth: int = 0) -> int:
    """Max nesting depth of the AST (proxy for structural complexity)."""
    if not node.children:
        return depth
    return max(_max_depth(child, depth + 1) for child in node.children)


def extract_ast_features(code: str, lang: str) -> dict:
    """
    Parse `code` with tree-sitter and derive structural metrics:
      - cyclomatic_complexity (approx, via decision-node count + 1)
      - max_nesting_depth
      - num_functions
      - num_calls
      - num_ast_nodes
      - avg_function_length (in nodes, rough proxy)
    Falls back to zeros if parsing fails (e.g. unsupported language or
    syntactically broken snippet — common in vuln datasets with partial files).
    """

    lang = lang.lower().strip()
    decision_types = _DECISION_NODE_TYPES.get(lang, set())
    function_types = _FUNCTION_NODE_TYPES.get(lang, set())
    call_types = _CALL_NODE_TYPES.get(lang, set())

    feats = {
        "cyclomatic_complexity": 0,
        "max_nesting_depth": 0,
        "num_functions": 0,                                             #to set default values to 0 for all the features at first
        "num_calls": 0,
        "num_ast_nodes": 0,
        "has_parse_error": 0,
    }

    try:
        parser = _get_parser(lang)
        tree = parser.parse(bytes(code, "utf8"))                        #to encode and turn python snippet to machine readable bytes for the parser to process 
        root = tree.root_node

        decision_count = 0
        function_count = 0                                              #to initialize the counters to start fresh
        call_count = 0
        node_count = 0

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

        feats["cyclomatic_complexity"] = decision_count + 1                     #to calculate the number of decisions as higher complexity in code = higher security
        feats["max_nesting_depth"] = _max_depth(root)
        feats["num_functions"] = function_count
        feats["num_calls"] = call_count
        feats["num_ast_nodes"] = node_count

    except Exception:
        feats["has_parse_error"] = 1        # Unsupported language / parser load failure — don't crash the pipeline, just fall back to lexical-only features for this sample.
    return feats




#Lexical/statistical features

def extract_lexical_features(code: str) -> dict:
    """language-agnostic features from raw text + regex, no parser needed."""
    tokens = tokenize(code)
    n_tokens = len(tokens) or 1
    
    lines = code.splitlines()
    n_lines = len(lines) or 1
    
    risky_hits = RISKY_API.findall(code)
    safe_hits = SAFE_API.findall(code)
    validation_hits = VALIDATION_PATTERN.findall(code)
    branch_hits = BRANCH_NODE_PATTERN.findall(code)

    normalized = normalize_identifiers(tokens)
    n_unique_vars = len({t for t in normalized if re.match(r"^VAR\d+$", t)})   #to count the number of variables in the code

    return {
        "n_tokens": n_tokens,
        "n_lines": n_lines,
        "avg_line_length": sum(len(l) for l in lines) / n_lines,
        "n_risky_api": len(risky_hits),
        "n_safe_api": len(safe_hits),
        "n_validation_calls": len(validation_hits),
        "n_branches": len(branch_hits),
        "risky_to_safe_ratio": (len(risky_hits) + 1) / (len(safe_hits) + 1),                #calculate risky functions relative to safe functions 
        "risky_density": len(risky_hits) / n_tokens,                                        #Measures the proportion of overall tokens that are high-risk API calls.
        "validation_density": len(validation_hits) / n_tokens,                              #measures the porportion of healthy functions in the code
        "n_unique_identifiers": n_unique_vars,
        "identifier_diversity": n_unique_vars / n_tokens,                                   # variety of naming
        "max_line_length": max((len(l) for l in lines), default=0),
        "n_string_literals": sum(1 for t in tokens if t.startswith(('"', "'"))),            #Counts every token in the code that begins with a single or double quote (' or ")
    }

def extract_features(code: str, lang: str = "c") -> dict:
    """Combine lexical + AST features into one flat feature dict for a single sample."""
    feats = {}
    feats.update(extract_lexical_features(code))
    feats.update(extract_ast_features(code, lang))
    return feats

