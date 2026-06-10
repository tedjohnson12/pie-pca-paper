"""
Get inter-project dependencies to go in showyourwork.yml
"""

from pathlib import Path
import ast
import yaml
import paths
PROJECT_ROOT = paths.root
SCRIPTS_PATH = paths.scripts
if __name__=='__main__':
    module_map = {
        path.stem: str(path.relative_to(PROJECT_ROOT))
        for path in SCRIPTS_PATH.glob('*.py')
    }
    dependencies = {}
    for file in SCRIPTS_PATH.glob("*.py"):
        if file == Path(__file__): # Ignore this file
            continue
        tree = ast.parse(file.read_text())
        
        file_deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in module_map:
                        file_deps.add(module_map[module])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    module = node.module.split('.')[0]
                    if module in module_map:
                        file_deps.add(module_map[module])
        if len(file_deps) == 0:
            continue
        dependencies[str(file.relative_to(PROJECT_ROOT))] = sorted(file_deps)

    print(yaml.dump({'dependencies':dependencies},indent=2, sort_keys=True))