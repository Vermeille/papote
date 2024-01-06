from typing import Optional, Union, get_origin


class OptionsBase:
    """
    Inherit this class, add members with type annotations.
    it will automatically handle the parsing of command line arguments.
    """

    def __init__(self):
        for nm, val in self.__class__.__dict__.items():
            if nm.startswith('__'):
                continue
            setattr(self, nm, val)

    def parse(self, line):
        annotations = self.__annotations__
        cmd, *args = line.split()
        if cmd not in annotations:
            raise ValueError(f'Unknown option {cmd}')
        ty = annotations[cmd]
        if get_origin(ty) is Union:
            if len(args) == 0:
                args = [None]
                ty = lambda x: x
            else:
                ty = ty.__args__[0]
        print(ty, args)
        print(f'{cmd} = {ty(args[0])}')
        setattr(self, cmd, ty(args[0]))
