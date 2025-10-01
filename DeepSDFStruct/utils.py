import logging
import DeepSDFStruct


def configure_logging(level=logging.DEBUG, logfile=None):
    logger = logging.getLogger(DeepSDFStruct.__name__)
    print(f"deepsdfstruct name: {DeepSDFStruct.__name__}")
    logger.setLevel(logging.DEBUG)

    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


_TUWIEN_COLOR_SCHEME = {
    "blue": (0, 102, 153),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue_1": (84, 133, 171),
    "blue_2": (114, 173, 213),
    "blue_3": (166, 213, 236),
    "blue_4": (223, 242, 253),
    "grey": (100, 99, 99),
    "grey_1": (157, 157, 156),
    "grey_2": (208, 208, 208),
    "grey_3": (237, 237, 237),
    "green": (0, 126, 113),
    "green_1": (106, 170, 165),
    "green_2": (162, 198, 194),
    "green_3": (233, 241, 240),
    "magenta": (186, 70, 130),
    "magenta_1": (205, 129, 168),
    "magenta_2": (223, 175, 202),
    "magenta_3": (245, 229, 239),
    "yellow": (225, 137, 34),
    "yellow_1": (238, 180, 115),
    "yellow_2": (245, 208, 168),
    "yellow_3": (153, 239, 225),
}
