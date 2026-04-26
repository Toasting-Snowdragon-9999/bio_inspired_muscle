import os, sys 
from enum import Enum
from datetime import datetime

class LogLvl(Enum):
    INFO = 0
    DEBUG = 1
    ERROR = 2
    CONSOLE = 4 # This is for printing in the app console

class Logger:
    def __init__(self, log_level: LogLvl = LogLvl.INFO, log_file: str = "app.log", log_to_file: bool = True):
        self.log_file = log_file
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.clear_on_init = True
        self.filename_width = 15
        self.handler = None  # Optional callback for CONSOLE logs

    def log(self, message: str, level: LogLvl = LogLvl.INFO):
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
        """Set a callback function to receive CONSOLE log messages (for display in the app's GUI console)."""
        self.handler = handler

    def _get_timestamp(self) -> str:
        now = datetime.now()
        return f"{now.day}/{now.month} - {now.hour:02}:{now.minute:02}"

    def _get_filename(self) -> str:
        if sys.argv and sys.argv[0]:
            return os.path.basename(sys.argv[0])
        return "unknown"

    def make_string(self, message: str, level: LogLvl = LogLvl.INFO) -> str:
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
        self.log(message, LogLvl.INFO)

    def debug(self, message: str):
        self.log(message, LogLvl.DEBUG)

    def error(self, message: str):
        self.log(message, LogLvl.ERROR)

    def console(self, message: str):
        self.log(message, LogLvl.CONSOLE)
    

