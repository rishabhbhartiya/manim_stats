#!/usr/bin/env python3
import ast
import os
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    name: str
    params: List[str]
    variables: List[str] = field(default_factory=list)

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)

@dataclass
class ModuleInfo:
    filepath: str
    classes: List[ClassInfo]
    functions: List[FunctionInfo]


class APIExtractor(ast.NodeVisitor):
    def __init__(self):
        self.module_info = ModuleInfo(filepath="", classes=[], functions=[])
        self.current_class = None

    def visit_ClassDef(self, node):
        prev = self.current_class
        class_info = ClassInfo(
            name=node.name,
            bases=[self._name(b) for b in node.bases]
        )
        self.current_class = class_info
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_FunctionDef(item)
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        class_info.class_variables.append(t.id)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    class_info.class_variables.append(item.target.id)
        self.module_info.classes.append(class_info)
        self.current_class = prev

    def visit_FunctionDef(self, node):
        params = []
        for arg in node.args.args:
            p = arg.arg
            if arg.annotation:
                p += f": {self._name(arg.annotation)}"
            params.append(p)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")

        # defaults
        defs = node.args.defaults
        if defs:
            for i, d in enumerate(defs):
                idx = len(params) - len(defs) + i
                if idx < len(params) and '=' not in params[idx]:
                    params[idx] += f"={ast.unparse(d)}"

        # local variables (non-self)
        variables = []
        seen = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id not in seen:
                        seen.add(t.id)
                        variables.append(t.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                if n.target.id not in seen:
                    seen.add(n.target.id)
                    variables.append(n.target.id)

        func = FunctionInfo(name=node.name, params=params, variables=variables)
        if self.current_class:
            self.current_class.methods.append(func)
        else:
            self.module_info.functions.append(func)

    def _name(self, node) -> str:
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return f"{self._name(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript): return f"{self._name(node.value)}[{self._name(node.slice)}]"
        try: return ast.unparse(node)
        except: return ""


def extract_file(filepath: str) -> ModuleInfo:
    with open(filepath, 'r', encoding='utf-8') as f:
        try: tree = ast.parse(f.read())
        except SyntaxError: return None
    ex = APIExtractor()
    ex.module_info.filepath = filepath
    ex.visit(tree)
    return ex.module_info


def extract_dir(root: str, exclude: Set[str] = None) -> Dict[str, ModuleInfo]:
    exclude = exclude or {'__pycache__', '.git', 'venv', 'env', '.venv'}
    modules = {}
    root_path = Path(root)
    for f in root_path.rglob("*.py"):
        if any(e in f.parts for e in exclude): continue
        rel = str(f.relative_to(root_path))
        info = extract_file(str(f))
        if info: modules[rel] = info
    return modules


def generate(modules: Dict[str, ModuleInfo]) -> str:
    lines = ["# API Reference\n"]
    for path, mod in sorted(modules.items()):
        lines.append(f"\n## {path}")
        for cls in mod.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"  class {cls.name}{bases}: {', '.join(m.name for m in cls.methods)}")
        for fn in mod.functions:
            lines.append(f"  def {fn.name}({', '.join(fn.params)})")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", default="API_REFERENCE.md")
    args = parser.parse_args()

    modules = extract_dir(args.directory)
    md = generate(modules)
    with open(args.output, 'w') as f:
        f.write(md)
    print(f"Done — {len(md.splitlines())} lines → {args.output}")