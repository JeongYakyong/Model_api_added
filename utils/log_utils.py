"""
Streamlit-aware logging utility.

Usage in pages:
    from utils.log_utils import st_log_status, render_log_sidebar_toggle

    with st_log_status("데이터 수집 중..."):
        daily_historical_update(start, end)

    # In sidebar (call once per page, after sidebar menu):
    render_log_sidebar_toggle()

Logs are displayed inline via st.status() during execution (auto-collapses
on completion), and buffered to st.session_state for the sidebar
mini-terminal history viewer.
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


class _InlineHandler(logging.Handler):
    """logging.Handler that writes log lines into a Streamlit container (st.status)."""

    def __init__(self, container):
        super().__init__()
        self._container = container

    def emit(self, record):
        if not hasattr(threading.current_thread(), "streamlit_script_run_ctx"):
            return  # worker thread — writing to the container would raise "missing ScriptRunContext"
        try:
            msg = self.format(record)
            ts = datetime.now().strftime("%H:%M:%S")
            self._container.text(f"[{ts}] {record.levelname:7s} | {msg}")
        except Exception:
            pass  # closed container


def _attach_handler():
    """Create and attach a _BufferHandler, removing any stale ones first."""
    handler = _BufferHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
    for lgr in loggers:
        # Remove stale _BufferHandlers left from interrupted Streamlit reruns
        for h in lgr.handlers[:]:
            if isinstance(h, _BufferHandler):
                lgr.removeHandler(h)
        lgr.addHandler(handler)

    return loggers, handler


def _detach_handler(loggers, handler):
    """Remove the handler from all loggers."""
    for lgr in loggers:
        lgr.removeHandler(handler)


@contextmanager
def st_log_status(label="실행 중...", done_label="완료"):
    """
    Context manager: captures logging output to the session-state buffer.

    When the sidebar '실시간 로그' checkbox is ON, logs also stream inline
    via st.status() (auto-collapses on completion). When OFF, a plain
    st.spinner() is shown instead.

    Args:
        label: status text shown while running
        done_label: text shown on completion
    """
    _append_log("INFO", f"── {label} ──")

    inline_enabled = st.session_state.get('_inline_log_enabled', False)

    if inline_enabled:
        with st.status(label, expanded=True) as status:
            buf_handler = _BufferHandler()
            buf_handler.setFormatter(logging.Formatter("%(message)s"))

            inline_handler = _InlineHandler(status)
            inline_handler.setFormatter(logging.Formatter("%(message)s"))

            loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
            for lgr in loggers:
                for h in lgr.handlers[:]:
                    if isinstance(h, (_BufferHandler, _InlineHandler)):
                        lgr.removeHandler(h)
                lgr.addHandler(buf_handler)
                lgr.addHandler(inline_handler)

            try:
                yield
            except Exception:
                _append_log("ERROR", f"── {label} - 오류 발생 ──")
                status.update(label=f"❌ {label} - 오류 발생", state="error", expanded=True)
                raise
            else:
                _append_log("INFO", f"── {done_label} ──")
                status.update(label=f"✅ {done_label}", state="complete", expanded=False)
            finally:
                for lgr in loggers:
                    lgr.removeHandler(buf_handler)
                    lgr.removeHandler(inline_handler)
    else:
        loggers, handler = _attach_handler()
        try:
            with st.spinner(label):
                yield
        except Exception:
            _append_log("ERROR", f"── {label} - 오류 발생 ──")
            raise
        else:
            _append_log("INFO", f"── {done_label} ──")
        finally:
            _detach_handler(loggers, handler)


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
    loggers, handler = _attach_handler()
    try:
        yield
    finally:
        _detach_handler(loggers, handler)


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


@st.dialog("📟 로그 히스토리", width="large")
def _show_log_dialog():
    """Dialog popup showing full log history in terminal style."""
    buf = _get_log_buffer()
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


def render_log_sidebar_toggle():
    """Render log controls in sidebar: inline log toggle + history dialog button."""
    if '_inline_log_enabled' not in st.session_state:
        st.session_state['_inline_log_enabled'] = False

    is_on = st.session_state['_inline_log_enabled']
    st.sidebar.caption("로그설정")
    if st.sidebar.button(
        "📟 실시간 로그 ON" if is_on else "📟 실시간 로그 OFF",
        type="primary" if is_on else "secondary",
        use_container_width=True,
        help="활성화 시 API 호출 결과를 실시간으로 표시합니다.",
    ):
        st.session_state['_inline_log_enabled'] = not is_on
        st.rerun()
    buf = _get_log_buffer()
    count = f" ({len(buf)})" if buf else ""
    if st.sidebar.button(f"📟 로그 히스토리{count}", use_container_width=True):
        _show_log_dialog()


def render_log_viewer():
    """Legacy no-op — log viewer is now a dialog from the sidebar."""
    pass
