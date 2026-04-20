"""SLMFS CLI entry point.

Usage:
    python -m slmfs init <file.md> [...]     Offline migration
    python -m slmfs add <file.md> [...]      Online ingestion
    python -m slmfs fuse [--mount=path]      Mount FUSE filesystem
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "init":
        from .init import main as init_main
        init_main()
    elif command == "add":
        from .add import main as add_main
        add_main()
    elif command == "fuse":
        from .fuse_layer import main as fuse_main
        fuse_main()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
