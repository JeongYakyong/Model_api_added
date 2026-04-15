"""
Streamlit-aware logging utility.

Usage in pages:
    from utils.log_utils import st_log_status, render_log_viewer

    with st_log_status("데이터 수집 중..."):
        daily_historical_update(start, end)

    # In sidebar (call once per page, after sidebar menu):
    render_log_viewer()

Logs are buffered to st.session_state and displayed in a sidebar
expander styled as a mini-terminal. The expander opens/closes
client-side without triggering a Streamlit rerun.
"""
import logging
import threading
from datetime import datetime
import streamlit as st
from contextlib import contextmanager

# Module-level loggers used across the project
LOGGER_NAMES = ['jejucr.api', 'jejucr.pipeline', 'jejucr.db']

# Session state key for the log buffer
_LOG_BUFFER_KEY = '_log_buffer'
_MAX_LOG_LINES = 300


def _get_log_buffer():
    """Get or initialize the persistent log buffer in session state."""
    if _LOG_BUFFER_KEY not in st.session_state:
        st.session_state[_LOG_BUFFER_KEY] = []
    return st.session_state[_LOG_BUFFER_KEY]


def _append_log(level_name, message):
    """Append a log entry to the session-state buffer."""
    if not hasattr(threading.current_thread(), "streamlit_script_run_ctx"):
        return  # worker thread — no Streamlit context available
    buf = _get_log_buffer()
    ts = datetime.now().strftime("%H:%M:%S")
    buf.append(f"[{ts}] {level_name:7s} | {message}")
    if len(buf) > _MAX_LOG_LINES:
        st.session_state[_LOG_BUFFER_KEY] = buf[-_MAX_LOG_LINES:]


class _BufferHandler(logging.Handler):
    """logging.Handler that writes only to the session-state buffer."""

    def emit(self, record):
        try:
            msg = self.format(record)
            _append_log(record.levelname, msg)
        except Exception:
            self.handleError(record)


@contextmanager
def st_log_status(label="실행 중...", done_label="완료"):
    """
    Context manager: shows st.spinner() for visual feedback and
    captures all logging output into the persistent session-state buffer.

    No inline log display — logs are only viewable via the sidebar
    mini-terminal (render_log_viewer).

    Args:
        label: spinner text shown while running
        done_label: text logged on completion
    """
    _append_log("INFO", f"── {label} ──")

    handler = _BufferHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
    for lgr in loggers:
        lgr.addHandler(handler)

    try:
        with st.spinner(label):
            yield
    except Exception:
        _append_log("ERROR", f"── {label} - 오류 발생 ──")
        raise
    else:
        _append_log("INFO", f"── {done_label} ──")
    finally:
        for lgr in loggers:
            lgr.removeHandler(handler)


@contextmanager
def log_capture():
    """
    Lightweight context manager: captures logs to the buffer only.
    No spinner, no UI — use with your own st.spinner() per step.

    Usage:
        with log_capture():
            with st.spinner("① KPX 수집 중..."):
                daily_historical_kpx(...)
            with st.spinner("② KMA 수집 중..."):
                daily_historical_kma(...)
    """
    handler = _BufferHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
    for lgr in loggers:
        lgr.addHandler(handler)
    try:
        yield
    finally:
        for lgr in loggers:
            lgr.removeHandler(handler)


# ============================================================================
# Mini-terminal log viewer (sidebar expander — no rerun on toggle)
# ============================================================================

_TERMINAL_CSS = """
<style>
.log-terminal {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.5;
    padding: 12px 16px;
    border-radius: 8px;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
.log-terminal .log-info    { color: #9cdcfe; }
.log-terminal .log-warning { color: #dcdcaa; }
.log-terminal .log-error   { color: #f44747; }
.log-terminal .log-debug   { color: #6a9955; }
.log-terminal .log-sep     { color: #569cd6; }
</style>
"""


def _colorize_line(line):
    """Wrap a log line in a span with the appropriate color class."""
    if "ERROR" in line:
        return f'<span class="log-error">{line}</span>'
    elif "WARNING" in line:
        return f'<span class="log-warning">{line}</span>'
    elif "DEBUG" in line:
        return f'<span class="log-debug">{line}</span>'
    elif "──" in line:
        return f'<span class="log-sep">{line}</span>'
    else:
        return f'<span class="log-info">{line}</span>'


def render_log_sidebar_toggle():
    """Render a small checkbox in the sidebar to toggle log viewer visibility."""
    buf = _get_log_buffer()
    count = f" ({len(buf)})" if buf else ""
    st.sidebar.checkbox(f"📟 로그{count}", key="_log_visible", value=False)


def render_log_viewer():
    """
    Render a mini-terminal log viewer as a main-area expander.
    Only shown when the sidebar checkbox is checked.
    """
    if not st.session_state.get("_log_visible", False):
        return

    buf = _get_log_buffer()
    label = f"📟 실행 로그 ({len(buf)}건)" if buf else "📟 실행 로그"

    with st.expander(label, expanded=True):
        if not buf:
            st.caption("아직 로그가 없습니다. 데이터 수집이나 예측을 실행하면 여기에 표시됩니다.")
            return

        colored_lines = [_colorize_line(line) for line in buf]
        html = (
            _TERMINAL_CSS
            + '<div class="log-terminal">'
            + "\n".join(colored_lines)
            + "</div>"
        )
        st.html(html)

        if st.button("🗑️ 로그 지우기", key="_log_clear_btn", use_container_width=True):
            st.session_state[_LOG_BUFFER_KEY] = []
            st.rerun()
