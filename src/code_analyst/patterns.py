"""Design pattern detection using AST heuristics."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detect common design patterns in Python source code using AST analysis."""

    def detect_patterns(self, source_code: str) -> list[dict[str, Any]]:
        """Analyze source code for design patterns. Returns list of detected patterns."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            logger.warning("Failed to parse source code: %s", exc)
            return []

        patterns: list[dict[str, Any]] = []
        patterns.extend(self._detect_singleton(tree))
        patterns.extend(self._detect_factory(tree))
        patterns.extend(self._detect_strategy(tree))
        patterns.extend(self._detect_observer(tree))
        patterns.extend(self._detect_decorator_pattern(tree))
        patterns.extend(self._detect_dependency_injection(tree))
        return patterns

    def _detect_singleton(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Singleton pattern: class with __new__ that caches instance, or _instance attribute."""
        results = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            has_new = False
            has_instance_attr = False
            for item in node.body:
                # Check for __new__ method
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__new__":
                    has_new = True
                # Check for _instance class variable
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id in ("_instance", "_instances"):
                            has_instance_attr = True
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id in ("_instance", "_instances"):
                        has_instance_attr = True

            if has_new or has_instance_attr:
                confidence = 0.9 if (has_new and has_instance_attr) else 0.7
                results.append({
                    "pattern": "Singleton",
                    "confidence": confidence,
                    "location": f"class {node.name} (line {node.lineno})",
                    "description": f"Class '{node.name}' appears to implement the Singleton pattern"
                    + (" via __new__ and _instance attribute." if has_new and has_instance_attr
                       else " via __new__ override." if has_new
                       else " via _instance class attribute."),
                })
        return results

    def _detect_factory(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Factory pattern: methods named create_*, build_*, or classes ending in Factory."""
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Factory"):
                results.append({
                    "pattern": "Factory",
                    "confidence": 0.8,
                    "location": f"class {node.name} (line {node.lineno})",
                    "description": f"Class '{node.name}' follows the Factory naming pattern.",
                })
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("create_") or node.name.startswith("build_"):
                    results.append({
                        "pattern": "Factory",
                        "confidence": 0.6,
                        "location": f"function {node.name} (line {node.lineno})",
                        "description": f"Function '{node.name}' may be a factory method.",
                    })
        return results

    def _detect_strategy(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Strategy pattern: abstract base classes with multiple concrete implementations."""
        results = []
        # Look for classes with abstract methods (ABC pattern)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if the class inherits from ABC or has abstractmethod decorators
            is_abstract = False
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in ("ABC", "Protocol"):
                    is_abstract = True
            if not is_abstract:
                continue
            # Check for abstract methods
            abstract_methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in item.decorator_list:
                        dec_name = ""
                        if isinstance(dec, ast.Name):
                            dec_name = dec.id
                        elif isinstance(dec, ast.Attribute):
                            dec_name = dec.attr
                        if dec_name == "abstractmethod":
                            abstract_methods.append(item.name)
            if abstract_methods:
                results.append({
                    "pattern": "Strategy",
                    "confidence": 0.7,
                    "location": f"class {node.name} (line {node.lineno})",
                    "description": (
                        f"Class '{node.name}' defines abstract methods {abstract_methods}, "
                        "suggesting a Strategy or Template Method pattern."
                    ),
                })
        return results

    def _detect_observer(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Observer pattern: classes with subscribe/notify/on_event methods or _observers/_listeners."""
        results = []
        observer_methods = {"subscribe", "unsubscribe", "notify", "add_listener", "remove_listener",
                            "add_observer", "remove_observer", "on_event", "emit"}
        observer_attrs = {"_observers", "_listeners", "_subscribers", "_callbacks", "_handlers"}

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            found_methods: list[str] = []
            found_attrs: list[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in observer_methods:
                        found_methods.append(item.name)
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id in observer_attrs:
                            found_attrs.append(target.id)
            if len(found_methods) >= 2 or (found_methods and found_attrs):
                results.append({
                    "pattern": "Observer",
                    "confidence": 0.75,
                    "location": f"class {node.name} (line {node.lineno})",
                    "description": (
                        f"Class '{node.name}' has observer-like methods {found_methods}"
                        + (f" and attributes {found_attrs}" if found_attrs else "")
                        + "."
                    ),
                })
        return results

    def _detect_decorator_pattern(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Decorator pattern: class wrapping another object of same interface."""
        results = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.endswith("Decorator") or node.name.endswith("Wrapper"):
                results.append({
                    "pattern": "Decorator",
                    "confidence": 0.7,
                    "location": f"class {node.name} (line {node.lineno})",
                    "description": f"Class '{node.name}' follows the Decorator/Wrapper naming convention.",
                })
                continue
            # Check __init__ for storing a wrapped object that shares a base class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Attribute) and isinstance(stmt.value, ast.Name):
                            if stmt.value.id == "self" and stmt.attr in ("_wrapped", "_inner", "_delegate", "_component"):
                                results.append({
                                    "pattern": "Decorator",
                                    "confidence": 0.6,
                                    "location": f"class {node.name} (line {node.lineno})",
                                    "description": (
                                        f"Class '{node.name}' stores a wrapped component "
                                        f"via self.{stmt.attr}, suggesting a Decorator pattern."
                                    ),
                                })
                                break
        return results

    def _detect_dependency_injection(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Detect Dependency Injection: __init__ accepting interface/abstract typed params."""
        results = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__"):
                    continue
                injected_params = []
                for arg in item.args.args:
                    if arg.arg == "self":
                        continue
                    if arg.annotation:
                        # If the annotation references a known abstract/interface name
                        ann_name = ""
                        if isinstance(arg.annotation, ast.Name):
                            ann_name = arg.annotation.id
                        elif isinstance(arg.annotation, ast.Attribute):
                            ann_name = arg.annotation.attr
                        if ann_name and not ann_name[0].islower():
                            # Non-primitive type hint suggests DI
                            injected_params.append(arg.arg)
                if len(injected_params) >= 2:
                    results.append({
                        "pattern": "Dependency Injection",
                        "confidence": 0.6,
                        "location": f"class {node.name} (line {node.lineno})",
                        "description": (
                            f"Class '{node.name}' accepts {len(injected_params)} typed dependencies "
                            f"in __init__: {injected_params}."
                        ),
                    })
        return results
