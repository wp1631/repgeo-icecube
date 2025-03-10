import logging
import colorama
from colorama import Fore, Style
from cv2 import log

# Initialize colorama for Windows compatibility
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,  # Light Black for DEBUG
        logging.INFO: Fore.BLUE,  # Blue for INFO
        logging.WARNING: Fore.YELLOW,  # Orange (Yellow) for WARNING
        logging.ERROR: Fore.RED,  # Red for ERROR
        logging.CRITICAL: Fore.MAGENTA,  # Magenta for CRITICAL
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        log_fmt = (
            f"{log_color}{record.levelname}: {record.getMessage()}{Style.RESET_ALL}"
        )
        return log_fmt


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Default level can be changed
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
