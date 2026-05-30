from . import medical, traffic, agriculture, industrial, aerial

DOMAINS = {
    "medical":     medical,
    "traffic":     traffic,
    "agriculture": agriculture,
    "industrial":  industrial,
    "aerial":      aerial,
}

__all__ = ["DOMAINS"]
