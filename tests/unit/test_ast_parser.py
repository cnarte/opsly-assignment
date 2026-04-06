"""Unit tests for ASTParser."""

from __future__ import annotations

import textwrap

import pytest

from src.indexer.ast_parser import ASTParser


@pytest.fixture()
def parser() -> ASTParser:
    return ASTParser(repo_id="test-repo")


# -- class extraction --------------------------------------------------------


class TestClassExtraction:
    def test_simple_class(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            class Foo:
                """A Foo class."""
                pass
        ''')
        result = parser.parse(src, "mod.py")
        assert len(result["classes"]) == 1
        cls = result["classes"][0]
        assert cls["name"] == "Foo"
        assert cls["kind"] == "class"
        assert cls["docstring"] == "A Foo class."
        assert cls["start_line"] == 1
        assert cls["bases"] == []

    def test_class_with_inheritance(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            class Child(Base, Mixin):
                pass
        ''')
        result = parser.parse(src, "mod.py")
        cls = result["classes"][0]
        assert cls["bases"] == ["Base", "Mixin"]

    def test_class_with_decorators(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            @dataclass
            @some_decorator(arg=1)
            class MyModel:
                pass
        ''')
        result = parser.parse(src, "mod.py")
        cls = result["classes"][0]
        assert "dataclass" in cls["decorators"]
        assert "some_decorator" in cls["decorators"]

    def test_class_with_docstring(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            class Documented:
                """This is the docstring."""
                x = 1
        ''')
        result = parser.parse(src, "mod.py")
        assert result["classes"][0]["docstring"] == "This is the docstring."


# -- function extraction -----------------------------------------------------


class TestFunctionExtraction:
    def test_simple_function(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            def greet(name: str) -> str:
                """Say hello."""
                return f"Hello, {name}"
        ''')
        result = parser.parse(src, "mod.py")
        assert len(result["functions"]) == 1
        fn = result["functions"][0]
        assert fn["name"] == "greet"
        assert fn["kind"] == "function"
        assert fn["docstring"] == "Say hello."
        assert fn["return_type"] == "str"
        assert fn["args"][0]["name"] == "name"
        assert fn["args"][0]["annotation"] == "str"

    def test_async_function(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            async def fetch(url: str) -> bytes:
                pass
        ''')
        result = parser.parse(src, "mod.py")
        fn = result["functions"][0]
        assert fn["is_async"] is True

    def test_function_with_decorators(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            @app.get("/items")
            def list_items():
                pass
        ''')
        result = parser.parse(src, "mod.py")
        fn = result["functions"][0]
        assert len(fn["decorators"]) == 1
        assert "app.get" in fn["decorators"][0]

    def test_typed_parameters(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            def process(a: int, b: str = "", *args: Any, key: bool = True, **kwargs: Any) -> None:
                pass
        ''')
        result = parser.parse(src, "mod.py")
        fn = result["functions"][0]
        names = [a["name"] for a in fn["args"]]
        assert "a" in names
        assert "b" in names
        assert "args" in names
        assert "key" in names
        assert "kwargs" in names


# -- method vs function ------------------------------------------------------


class TestMethodDetection:
    def test_method_inside_class(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            class MyClass:
                def my_method(self) -> None:
                    pass
        ''')
        result = parser.parse(src, "mod.py")
        methods = [f for f in result["functions"] if f["kind"] == "method"]
        functions = [f for f in result["functions"] if f["kind"] == "function"]
        assert len(methods) == 1
        assert methods[0]["name"] == "my_method"
        assert len(functions) == 0

    def test_standalone_function_not_method(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            def standalone():
                pass
        ''')
        result = parser.parse(src, "mod.py")
        assert result["functions"][0]["kind"] == "function"


# -- nested functions --------------------------------------------------------


class TestNestedFunctions:
    def test_nested_function_distinct_symbol_id(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            def outer():
                def inner():
                    pass
        ''')
        result = parser.parse(src, "mod.py")
        fns = result["functions"]
        assert len(fns) == 2
        ids = {f["symbol_id"] for f in fns}
        assert len(ids) == 2  # distinct symbol_ids

    def test_nested_scope_chain(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            def outer():
                def inner():
                    pass
        ''')
        result = parser.parse(src, "mod.py")
        inner = [f for f in result["functions"] if f["name"] == "inner"][0]
        assert "outer" in inner["scope_chain"]


# -- import extraction -------------------------------------------------------


class TestImportExtraction:
    def test_import_statement(self, parser: ASTParser) -> None:
        src = "import os\nimport sys\n"
        result = parser.parse(src, "mod.py")
        assert len(result["imports"]) == 2
        names = {i["name"] for i in result["imports"]}
        assert names == {"os", "sys"}

    def test_from_import(self, parser: ASTParser) -> None:
        src = "from os.path import join, exists\n"
        result = parser.parse(src, "mod.py")
        assert len(result["imports"]) == 2
        assert result["imports"][0]["module"] == "os.path"
        assert result["imports"][0]["type"] == "from_import"

    def test_relative_import(self, parser: ASTParser) -> None:
        src = "from ..utils import helper\n"
        result = parser.parse(src, "mod.py")
        imp = result["imports"][0]
        assert imp["level"] == 2
        assert imp["module"] == "utils"
        assert imp["name"] == "helper"

    def test_alias(self, parser: ASTParser) -> None:
        src = "import numpy as np\n"
        result = parser.parse(src, "mod.py")
        imp = result["imports"][0]
        assert imp["alias"] == "np"
        assert imp["name"] == "numpy"


# -- module-level features ---------------------------------------------------


class TestModuleLevel:
    def test_module_docstring(self, parser: ASTParser) -> None:
        src = '"""Module docstring."""\nx = 1\n'
        result = parser.parse(src, "mod.py")
        assert result["module_docstring"] == "Module docstring."

    def test_all_exports(self, parser: ASTParser) -> None:
        src = '__all__ = ["Foo", "bar"]\n'
        result = parser.parse(src, "mod.py")
        assert result["all_exports"] == ["Foo", "bar"]

    def test_no_all(self, parser: ASTParser) -> None:
        src = "x = 1\n"
        result = parser.parse(src, "mod.py")
        assert result["all_exports"] is None


# -- symbol_id ---------------------------------------------------------------


class TestSymbolId:
    def test_build_symbol_id_format(self) -> None:
        sid = ASTParser.build_symbol_id("repo", "class", "<module>", "Foo", 0)
        assert sid == "repo:class:<module>:Foo:0"

    def test_unique_ids_same_name_different_scope(self, parser: ASTParser) -> None:
        src = textwrap.dedent('''\
            class A:
                def run(self):
                    pass
            class B:
                def run(self):
                    pass
        ''')
        result = parser.parse(src, "mod.py")
        methods = [f for f in result["functions"] if f["name"] == "run"]
        assert len(methods) == 2
        assert methods[0]["symbol_id"] != methods[1]["symbol_id"]


# -- syntax error handling ---------------------------------------------------


class TestErrorHandling:
    def test_syntax_error_returns_error(self, parser: ASTParser) -> None:
        result = parser.parse("def bad(:", "bad.py")
        assert "error" in result
        assert result["entities"] == {}
