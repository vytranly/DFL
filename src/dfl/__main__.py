"""`python -m dfl`: chuyển cho dfl.cli. Theo tài liệu, __main__.py của gói giữ ngắn và không rào bằng
`if __name__ == "__main__"`: https://docs.python.org/3.12/library/__main__.html#main-py-in-python-packages
"""

from dfl.cli import main

raise SystemExit(main())
