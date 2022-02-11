import enum


class AutoNum(enum.Enum):
    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


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
    shoes = ()
    trash = ()
    white_glass = ()

    @classmethod
    def choices(cls):
        return tuple((i.name, i.value) for i in cls)

    @classmethod
    def to_list(cls):
        return list(i.name for i in cls)


class Area(AutoNum):
    Ajax = ()
    Aurora = ()
    Brampton = ()
    Brock = ()
    Burlington = ()
    Caledon = ()
    Clarington = ()
    East_Gwillimbury = ()
    Georgina = ()
    Halton_Hills = ()
    King = ()
    Markham = ()
    Milton = ()
    Mississauga = ()
    Newmarket = ()
    Oakville = ()
    Oshawa = ()
    Pickering = ()
    Richmond_Hill = ()
    Scugog = ()
    Toronto = ()
    Uxbridge = ()
    Vaughan = ()
    Whitby = ()
    Whitchurch_Stouffville = ()

    @classmethod
    def choices(cls):
        return tuple((i.name, i.value) for i in cls)

