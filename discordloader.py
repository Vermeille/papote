def keep(line):
    """Returns false if the line starts with a {, [, a link, or is empty"""
    return not (line.startswith('{') or line.startswith('[')
                or line.startswith('http') or line == '\n')


def discord_load(path):
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
    lines = ''.join([line for line in lines if keep(line)])
    return lines
