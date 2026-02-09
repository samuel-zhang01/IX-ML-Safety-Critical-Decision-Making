#!/usr/bin/env python3
"""
nb_helper.py - Jupyter Notebook cell manipulation helper.

Usage:
    python nb_helper.py list <notebook>
    python nb_helper.py insert <notebook> <after_index> <code|markdown> <content_file>
    python nb_helper.py insert_raw <notebook> <after_index> <code|markdown> <content_string>
    python nb_helper.py edit <notebook> <cell_index> <content_file>
    python nb_helper.py edit_raw <notebook> <cell_index> <content_string>
    python nb_helper.py set_type <notebook> <cell_index> <code|markdown>
    python nb_helper.py delete <notebook> <cell_index>
    python nb_helper.py count <notebook>
"""
import json
import sys
import os


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def make_cell(cell_type, source):
    """Create a new notebook cell dict."""
    if cell_type == 'code':
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else source.splitlines(True)
        }
    else:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else source.splitlines(True)
        }


def cmd_list(nb_path):
    nb = load_notebook(nb_path)
    cells = nb['cells']
    print(f"Total cells: {len(cells)}")
    print("-" * 80)
    for i, cell in enumerate(cells):
        ct = cell['cell_type']
        src = ''.join(cell.get('source', []))
        first_line = src.split('\n')[0][:60] if src.strip() else '<empty>'
        print(f"[{i:3d}] {ct:8s} | {first_line}")


def cmd_count(nb_path):
    nb = load_notebook(nb_path)
    print(len(nb['cells']))


def cmd_insert(nb_path, after_index, cell_type, content):
    nb = load_notebook(nb_path)
    cells = nb['cells']
    after_index = int(after_index)
    if after_index < -1 or after_index >= len(cells):
        print(f"Error: after_index {after_index} out of range (0-{len(cells)-1})")
        sys.exit(1)
    new_cell = make_cell(cell_type, content)
    cells.insert(after_index + 1, new_cell)
    save_notebook(nb, nb_path)
    print(f"Inserted {cell_type} cell at index {after_index + 1} (total: {len(cells)})")


def cmd_edit(nb_path, cell_index, content):
    nb = load_notebook(nb_path)
    cells = nb['cells']
    cell_index = int(cell_index)
    if cell_index < 0 or cell_index >= len(cells):
        print(f"Error: cell_index {cell_index} out of range (0-{len(cells)-1})")
        sys.exit(1)
    cells[cell_index]['source'] = content if isinstance(content, list) else content.splitlines(True)
    # Clear outputs for code cells
    if cells[cell_index]['cell_type'] == 'code':
        cells[cell_index]['outputs'] = []
        cells[cell_index]['execution_count'] = None
    save_notebook(nb, nb_path)
    print(f"Edited cell {cell_index}")


def cmd_set_type(nb_path, cell_index, cell_type):
    nb = load_notebook(nb_path)
    cells = nb['cells']
    cell_index = int(cell_index)
    if cell_index < 0 or cell_index >= len(cells):
        print(f"Error: cell_index {cell_index} out of range (0-{len(cells)-1})")
        sys.exit(1)
    old_source = cells[cell_index].get('source', [])
    cells[cell_index] = make_cell(cell_type, old_source)
    save_notebook(nb, nb_path)
    print(f"Set cell {cell_index} type to {cell_type}")


def cmd_delete(nb_path, cell_index):
    nb = load_notebook(nb_path)
    cells = nb['cells']
    cell_index = int(cell_index)
    if cell_index < 0 or cell_index >= len(cells):
        print(f"Error: cell_index {cell_index} out of range (0-{len(cells)-1})")
        sys.exit(1)
    cells.pop(cell_index)
    save_notebook(nb, nb_path)
    print(f"Deleted cell {cell_index} (total: {len(cells)})")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    nb_path = sys.argv[2]

    if cmd == 'list':
        cmd_list(nb_path)
    elif cmd == 'count':
        cmd_count(nb_path)
    elif cmd == 'insert':
        # insert <notebook> <after_index> <cell_type> <content_file>
        after_index = sys.argv[3]
        cell_type = sys.argv[4]
        content_file = sys.argv[5]
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        cmd_insert(nb_path, after_index, cell_type, content)
    elif cmd == 'insert_raw':
        # insert_raw <notebook> <after_index> <cell_type> <content_string>
        after_index = sys.argv[3]
        cell_type = sys.argv[4]
        content = sys.argv[5]
        cmd_insert(nb_path, after_index, cell_type, content)
    elif cmd == 'edit':
        # edit <notebook> <cell_index> <content_file>
        cell_index = sys.argv[3]
        content_file = sys.argv[4]
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        cmd_edit(nb_path, cell_index, content)
    elif cmd == 'edit_raw':
        # edit_raw <notebook> <cell_index> <content_string>
        cell_index = sys.argv[3]
        content = sys.argv[4]
        cmd_edit(nb_path, cell_index, content)
    elif cmd == 'set_type':
        cell_index = sys.argv[3]
        cell_type = sys.argv[4]
        cmd_set_type(nb_path, cell_index, cell_type)
    elif cmd == 'delete':
        cell_index = sys.argv[3]
        cmd_delete(nb_path, cell_index)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
