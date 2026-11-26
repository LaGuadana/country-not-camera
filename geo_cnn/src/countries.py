"""ISO 3166-1 alpha-2 code -> short English country name (via pycountry if installed)."""
_OVERRIDE = {"RU": "Russia", "KR": "South Korea", "TW": "Taiwan", "BO": "Bolivia",
             "VE": "Venezuela", "IR": "Iran", "TZ": "Tanzania", "VN": "Vietnam",
             "LA": "Laos", "SY": "Syria", "MD": "Moldova", "CZ": "Czechia",
             "GB": "United Kingdom", "US": "United States", "NA": "Namibia",
             "CD": "DR Congo", "CG": "Congo", "CI": "Ivory Coast", "BA": "Bosnia & Herz.",
             "KP": "North Korea", "PS": "Palestine", "MK": "North Macedonia", "TR": "Turkey"}


def name_of(code):
    if code in _OVERRIDE:
        return _OVERRIDE[code]
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code)
        if c is not None:
            return getattr(c, "common_name", None) or c.name
    except Exception:
        pass
    return code
