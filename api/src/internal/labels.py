import enum


class AutoNum(enum.Enum):
    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value


class Label(AutoNum):
    battery = ()
    biological = ()
    brown_glass = ()
    cardboard = ()
    clothes = ()
    green_class = ()
    metal = ()
    paper = ()
    plastic = ()
    trash = ()
    white_glass = ()
