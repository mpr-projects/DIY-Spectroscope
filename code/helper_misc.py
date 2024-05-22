


def px(*args):  # rounds and converts floats to ints
    res = [int(round(v)) for v in args]
    return res[0] if len(res) == 1 else res


def is_float(val):
    try:
        float(val)

    except ValueError:
        return False

    return True


def is_float_or_empty(val):
    if val == '':
        return True

    return is_float(val)
