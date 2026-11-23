"""Small text rules shared by the corpus facts and the locate step."""

# Token counts are estimated as characters over four.
CHARS_PER_TOKEN = 4

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# A written-out date, "March 14, 2016", with month, day and year as groups.
LONG_DATE_PATTERN = rf"\b({MONTHS})\s+(\d{{1,2}}),\s*((?:19|20)\d\d)\b"
