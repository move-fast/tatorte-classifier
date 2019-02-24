import logging
from configuration import LOG_FILE


def create_logger(name: str) -> logging.Logger:
    """Create a Logger, with filehandler, streamhandler and formatter
    
    Arguments:
        name {str} -- The name of the logger
    
    Returns:
        logging.Logger -- The logger with added Handlers
    """

    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
