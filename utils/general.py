import os


def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def get_data_root():
    return '/data4/ouyjb/cnnimageretrieval-pytorch-master/data'
    # return os.path.join(get_root(), 'data')


def htime(c):
    c = round(c)

    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)