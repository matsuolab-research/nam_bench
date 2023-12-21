from . import moving_digits
from . import moving_box


NAME2DATASETS_FN = {
    "MovingDigits": moving_digits.load,
    "MovingBox": moving_box.load,
}