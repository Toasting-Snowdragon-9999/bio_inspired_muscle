"""@brief Lightweight leveled logger that writes formatted messages to a file and/or the
console, with a dedicated CONSOLE level that can be routed to a GUI handler callback."""

import os, sys
from enum import Enum
from datetime import datetime

class LogLvl(Enum):
    """@brief Severity/destination levels for the Logger; CONSOLE is routed to the app GUI console."""
    INFO = 0
    DEBUG = 1
    ERROR = 2
    CONSOLE = 4 # This is for printing in the app console

class Logger:
    """@brief Leveled logger that formats messages and emits them to a file, stdout/stderr, or a GUI handler."""

    def __init__(self, log_level: LogLvl = LogLvl.INFO, log_file: str = "app.log", log_to_file: bool = True):
        """@brief Initialize the logger with a minimum level, target file, and file-output toggle.
        @param log_level: Minimum LogLvl that will be emitted (messages below this are dropped).
        @param log_file: Path of the file to append formatted log lines to.
        @param log_to_file: When True, write emitted messages to log_file."""
        self.log_file = log_file
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.clear_on_init = True
        self.filename_width = 15
        self.handler = None  # Optional callback for CONSOLE logs

    def log(self, message: str, level: LogLvl = LogLvl.INFO):
        """@brief Format and dispatch a message if its level meets the configured threshold.
        @param message: The text to log.
        @param level: Severity/destination LogLvl of this message; ERROR goes to stderr, CONSOLE to the handler."""
        if level.value >= self.log_level.value:
            formatted = self.make_string(message, level)

            if self.log_to_file:
                with open(self.log_file, "a") as f:
                    f.write(formatted + "\n")

            if level == LogLvl.ERROR:
                print(formatted, file=sys.stderr)

            elif level == LogLvl.CONSOLE:
                if self.handler is not None:
                    self.handler(formatted)
            else:
                print(formatted)
    
    def set_handler(self, handler):
        """@brief Register the callback used for CONSOLE-level messages.
        Set a callback function to receive CONSOLE log messages (for display in the app's GUI console).
        @param handler: Callable invoked with the formatted string for each CONSOLE log."""
        self.handler = handler

    def _get_timestamp(self) -> str:
        """@brief Build a short 'day/month - HH:MM' timestamp for the current local time.
        @return Formatted timestamp string."""
        now = datetime.now()
        return f"{now.day}/{now.month} - {now.hour:02}:{now.minute:02}"

    def _get_filename(self) -> str:
        """@brief Return the base name of the running script, or 'unknown' if unavailable.
        @return The script's basename, or 'unknown'."""
        if sys.argv and sys.argv[0]:
            return os.path.basename(sys.argv[0])
        return "unknown"

    def make_string(self, message: str, level: LogLvl = LogLvl.INFO) -> str:
        """@brief Compose the final log line, prefixing level, padded filename, and timestamp (raw for CONSOLE).
        @param message: The raw message text.
        @param level: LogLvl that determines formatting; CONSOLE returns the message unchanged.
        @return The fully formatted log string."""
        if level == LogLvl.CONSOLE:
            return message

        timestamp = self._get_timestamp()
        filename = self._get_filename()

        # Align filename column
        if len(filename) <= self.filename_width:
            filename_field = f"{filename:<{self.filename_width}}"
        else:
            # Let it overflow (your preference)
            filename_field = filename

        return f"[{level.name}] {filename_field} - {timestamp} {message}"
    
    def info(self, message: str):
        """@brief Log a message at INFO level.
        @param message: The text to log."""
        self.log(message, LogLvl.INFO)

    def debug(self, message: str):
        """@brief Log a message at DEBUG level.
        @param message: The text to log."""
        self.log(message, LogLvl.DEBUG)

    def error(self, message: str):
        """@brief Log a message at ERROR level (emitted to stderr).
        @param message: The text to log."""
        self.log(message, LogLvl.ERROR)

    def console(self, message: str):
        """@brief Log a message at CONSOLE level (routed to the registered GUI handler).
        @param message: The text to log."""
        self.log(message, LogLvl.CONSOLE)
    

