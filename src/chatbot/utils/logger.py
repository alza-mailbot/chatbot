"""Central application logger. All modules should import ``logger`` from here."""

import logging

from chatbot.config import get_settings

logger = logging.getLogger("chatbot")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
logger.addHandler(_handler)
logger.setLevel(get_settings().log_level)
