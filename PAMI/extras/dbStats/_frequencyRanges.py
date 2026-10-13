"""Shared equal-width distributions of positive item frequencies."""


def frequencyRanges(frequencies, numberOfRanges: int = 6) -> dict:
    """Return {inclusive upper frequency bound: number of items}.

    Ranges cover (0, max frequency]. Whole-number frequencies use ceiling
    integer bounds, with repeated bounds omitted. Empty ranges are retained.
    """
    if isinstance(numberOfRanges, bool) or not isinstance(numberOfRanges, int) or numberOfRanges <= 0:
        raise ValueError("numberOfRanges must be a positive int")

    values = list(frequencies)
    if not values:
        return {}

    maximum = max(values)
    integral = all(value == int(value) for value in values)
    if integral:
        maximum = int(maximum)

    result = {}
    previous = 0
    for index in range(1, numberOfRanges + 1):
        if index == numberOfRanges:
            upper = maximum
        elif integral:
            upper = (maximum * index + numberOfRanges - 1) // numberOfRanges
        else:
            upper = maximum * index / numberOfRanges
        if upper == previous:
            continue
        result[upper] = sum(previous < frequency <= upper for frequency in values)
        previous = upper
    return result
