import sys
from emmy_ptx.compiler import compile_file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python compile.py <kernel.ir.json>")

    compile_file(sys.argv[1])
