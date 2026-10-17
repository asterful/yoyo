import ast
import os

# ==========================================
# CONFIGURATION
# Automatically finds your package relative to .vscode/
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(
    script_dir, "..", "features"
)  # Change 'my_package' if your folder name is different


def generate_init():
  init_path = os.path.join(TARGET_DIR, "__init__.py")

  if not os.path.isdir(TARGET_DIR):
    print(f"Error: Target directory '{TARGET_DIR}' does not exist.")
    return

  import_lines = []

  # Find all python files, ignoring __init__.py and private files
  files = sorted([
      f
      for f in os.listdir(TARGET_DIR)
      if f.endswith(".py") and f != "__init__.py" and not f.startswith("_")
  ])

  for filename in files:
    mod_name = filename[:-3]
    filepath = os.path.join(TARGET_DIR, filename)

    # Read and parse the file into an Abstract Syntax Tree (AST)
    try:
      with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    except Exception as e:
      print(f"Skipping {filename} due to parse error: {e}")
      continue

    # Extract all top-level classes and functions that don't start with '_'
    symbols = []
    for node in tree.body:
      if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        if not node.name.startswith("_"):
          symbols.append(node.name)

    # If the file defines classes/functions, generate a flat import line
    if symbols:
      symbols = sorted(list(set(symbols)))
      symbols_str = ", ".join(symbols)
      import_lines.append(f"from .{mod_name} import {symbols_str}\n")

  # Overwrite __init__.py with the generated flat imports
  with open(init_path, "w", encoding="utf-8") as f:
    f.writelines(import_lines)

  print(f"[Auto-Init] Successfully updated {init_path} with flat imports!")


if __name__ == "__main__":
  generate_init()