"""
AST-based Python code parser for topology analysis.

Extracts structure, dependencies, and call relationships from Python source code.
"""

import ast
import os
from pathlib import Path
from typing import List, Optional, Set, Dict
from datetime import datetime

from llmops_monitoring.topology.models import (
    ModuleInfo,
    FunctionInfo,
    ClassInfo,
    ImportStatement,
    ImportType,
    FunctionType
)
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class CodeParser:
    """
    Parse Python source code using AST.

    Extracts:
    - Module structure
    - Import relationships
    - Function and class definitions
    - Function call relationships
    - Complexity metrics
    """

    def __init__(self):
        """Initialize parser."""
        self.current_module_path: str = ""
        self.current_file_path: str = ""

    def parse_file(self, file_path: str, module_path: Optional[str] = None) -> Optional[ModuleInfo]:
        """
        Parse a single Python file.

        Args:
            file_path: Path to Python file
            module_path: Dot-separated module path (auto-detected if None)

        Returns:
            ModuleInfo or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse AST
            tree = ast.parse(source, filename=file_path)

            # Auto-detect module path if not provided
            if module_path is None:
                module_path = self._path_to_module(file_path)

            self.current_module_path = module_path
            self.current_file_path = file_path

            # Extract module info
            module_info = ModuleInfo(
                name=module_path.split('.')[-1] if module_path else '',
                module_path=module_path,
                file_path=file_path
            )

            # Extract module docstring
            module_info.docstring = ast.get_docstring(tree)

            # Count lines
            module_info.line_count = len(source.splitlines())

            # Get last modified time
            module_info.last_modified = datetime.fromtimestamp(
                os.path.getmtime(file_path)
            )

            # Extract imports
            module_info.imports = self._extract_imports(tree)
            module_info.import_count = len(module_info.imports)

            # Extract dependencies from imports
            module_info.dependencies = self._extract_dependencies(module_info.imports)

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Only top-level functions (not methods)
                    if self._is_top_level(node, tree):
                        func_info = self._extract_function(node, None)
                        module_info.functions.append(func_info)

                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class(node)
                    module_info.classes.append(class_info)

            # Calculate complexity
            module_info.complexity_score = self._calculate_module_complexity(module_info)

            logger.debug(f"Parsed module: {module_path} ({len(module_info.functions)} functions, {len(module_info.classes)} classes)")

            return module_info

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def parse_repository(
        self,
        repo_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, ModuleInfo]:
        """
        Parse entire repository.

        Args:
            repo_path: Path to repository root
            include_patterns: Glob patterns to include (default: ["**/*.py"])
            exclude_patterns: Patterns to exclude (default: ["**/test_*.py", "**/__pycache__/**"])

        Returns:
            Dictionary mapping module_path -> ModuleInfo
        """
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        if exclude_patterns is None:
            exclude_patterns = [
                "**/test_*.py",
                "**/*_test.py",
                "**/tests/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/venv/**",
                "**/.git/**"
            ]

        modules = {}
        repo_path_obj = Path(repo_path)

        # Find all Python files
        python_files = []
        for pattern in include_patterns:
            python_files.extend(repo_path_obj.glob(pattern))

        # Filter excluded
        python_files = [
            f for f in python_files
            if not any(f.match(exclude) for exclude in exclude_patterns)
        ]

        logger.info(f"Found {len(python_files)} Python files in {repo_path}")

        # Parse each file
        for file_path in python_files:
            # Calculate module path relative to repo
            try:
                relative_path = file_path.relative_to(repo_path_obj)
                module_path = self._path_to_module(str(relative_path))

                module_info = self.parse_file(str(file_path), module_path)
                if module_info:
                    modules[module_path] = module_info

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Successfully parsed {len(modules)} modules")

        return modules

    def _extract_imports(self, tree: ast.AST) -> List[ImportStatement]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportStatement(
                        module_name=alias.name,
                        imported_names=[alias.name],
                        import_type=ImportType.IMPORT,
                        alias=alias.asname,
                        line_number=node.lineno
                    ))

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                imported_names = [alias.name for alias in node.names]

                import_type = ImportType.RELATIVE if node.level > 0 else ImportType.FROM_IMPORT

                imports.append(ImportStatement(
                    module_name=module_name,
                    imported_names=imported_names,
                    import_type=import_type,
                    level=node.level,
                    line_number=node.lineno
                ))

        return imports

    def _extract_dependencies(self, imports: List[ImportStatement]) -> Set[str]:
        """Extract unique module dependencies from imports."""
        dependencies = set()

        for imp in imports:
            if imp.module_name:
                # Get top-level package
                parts = imp.module_name.split('.')
                if parts:
                    dependencies.add(parts[0])

        return dependencies

    def _extract_function(
        self,
        node: ast.FunctionDef,
        parent_class: Optional[str]
    ) -> FunctionInfo:
        """Extract function information from AST node."""
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Determine function type
        if parent_class:
            func_type = FunctionType.ASYNC_METHOD if is_async else FunctionType.METHOD

            # Check for special decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id == 'property':
                        func_type = FunctionType.PROPERTY
                    elif decorator.id == 'staticmethod':
                        func_type = FunctionType.STATIC_METHOD
                    elif decorator.id == 'classmethod':
                        func_type = FunctionType.CLASS_METHOD
        else:
            func_type = FunctionType.ASYNC_FUNCTION if is_async else FunctionType.FUNCTION

        # Qualified name
        if parent_class:
            qualified_name = f"{self.current_module_path}.{parent_class}.{node.name}"
        else:
            qualified_name = f"{self.current_module_path}.{node.name}"

        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)

        # Extract return type (if annotated)
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else None

        # Extract function calls
        calls = self._extract_calls(node)

        # Calculate metrics
        line_count = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        complexity = self._calculate_complexity(node)

        func_info = FunctionInfo(
            name=node.name,
            qualified_name=qualified_name,
            function_type=func_type,
            file_path=self.current_file_path,
            line_start=node.lineno,
            line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            complexity=complexity,
            line_count=line_count,
            is_async=is_async,
            calls=calls,
            parent_class=parent_class
        )

        return func_info

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information from AST node."""
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function(item, node.name)
                methods.append(method_info)

        # Calculate metrics
        line_count = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0

        class_info = ClassInfo(
            name=node.name,
            qualified_name=f"{self.current_module_path}.{node.name}",
            file_path=self.current_file_path,
            line_start=node.lineno,
            line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            base_classes=base_classes,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            methods=methods,
            line_count=line_count,
            method_count=len(methods)
        )

        return class_info

    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from an AST node."""
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Try to get function name
                func_name = None

                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    func_name = child.func.attr
                elif isinstance(child.func, ast.Subscript):
                    # Handle callable subscripts
                    continue

                if func_name:
                    calls.append(func_name)

        return calls

    def _calculate_complexity(self, node: ast.AST) -> float:
        """
        Calculate cyclomatic complexity.

        Simple metric: count decision points + 1
        """
        complexity = 1.0

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            # Boolean operators add complexity
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # Comprehensions
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _calculate_module_complexity(self, module: ModuleInfo) -> float:
        """Calculate overall module complexity."""
        if not module.functions and not module.classes:
            return 0.0

        total_complexity = 0.0
        count = 0

        # Sum function complexities
        for func in module.functions:
            total_complexity += func.complexity
            count += 1

        # Sum method complexities
        for cls in module.classes:
            for method in cls.methods:
                total_complexity += method.complexity
                count += 1

        return total_complexity / count if count > 0 else 0.0

    def _is_top_level(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is at module top level (not nested in class)."""
        # Simple heuristic: check if any ClassDef contains this node
        for item in tree.body:
            if isinstance(item, ast.ClassDef):
                if node in ast.walk(item):
                    return False
        return True

    def _path_to_module(self, file_path: str) -> str:
        """
        Convert file path to Python module path.

        Examples:
            src/package/module.py -> src.package.module
            package/subpackage/module.py -> package.subpackage.module
        """
        # Remove .py extension
        if file_path.endswith('.py'):
            file_path = file_path[:-3]

        # Replace path separators with dots
        module_path = file_path.replace(os.sep, '.')
        module_path = module_path.replace('/', '.')  # Unix
        module_path = module_path.replace('\\', '.')  # Windows

        # Remove leading/trailing dots
        module_path = module_path.strip('.')

        # Handle __init__.py
        if module_path.endswith('.__init__'):
            module_path = module_path[:-9]

        return module_path
