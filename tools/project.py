import sys
sys.dont_write_bytecode = True
from workflow import main
if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
