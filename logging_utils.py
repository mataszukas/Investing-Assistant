import logging
import sys


class _MaxLevelFilter(logging.Filter):

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def setup_logging(level: str = 'INFO', name: str = 'etf_rag_bot') -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s -> %(message)s')
    h_out = logging.StreamHandler(sys.stdout)
    h_out.setFormatter(fmt)
    h_out.setLevel(logging.DEBUG)
    h_out.addFilter(_MaxLevelFilter(logging.WARNING))
    root.addHandler(h_out)

    h_err = logging.StreamHandler(sys.stderr)
    h_err.setFormatter(fmt)
    h_err.setLevel(logging.ERROR)
    root.addHandler(h_err)

    return logging.getLogger(name)


logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
