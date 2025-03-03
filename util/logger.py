import logging
import config

logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
)

log = logging.getLogger(__name__)
