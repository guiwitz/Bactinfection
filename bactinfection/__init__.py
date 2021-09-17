from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("bactinfection").version
except DistributionNotFound:
    # package is not installed
    pass