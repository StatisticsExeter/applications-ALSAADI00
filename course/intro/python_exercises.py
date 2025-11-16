from datetime import datetime


def sum_list(x):
    """Return the sum of a list x."""
    return sum(x)


def squares(x):
    """Return a list containing the squares of x."""
    return [i * i for i in x]


def evens(x):
    """Return a list containing only the even numbers in x."""
    return [i for i in x if i % 2 == 0]


def mean(x):
    """Return the mean of the list x."""
    return sum(x) / len(x)


def reverse_string(s):
    """Return the reverse of the string s."""
    return s[::-1]


def max_value(x):
    """Return the maximum value in the list x."""
    return max(x)


def filter_even(x):
    """Return a list containing only the even numbers in x."""
    return [i for i in x if i % 2 == 0]


def get_fifth_row(df):
    """Return the fifth row of the DataFrame df."""
    return df.iloc[4]


def column_mean(df, col_name):
    """Return the mean of the column col_name in the DataFrame df."""
    return df[col_name].mean()


def lookup_key(data, key, key_col=None, value_col=None):
    """
    Look up a key and return the corresponding value.

    If data is a dict or Series and key_col/value_col are None, return data[key].
    If data is a DataFrame and key_col and value_col are given, find the row where
    data[key_col] == key and return the value from value_col. If key is not found,
    return None.
    """
    if key_col is None and value_col is None:
        return data.get(key, None)

    rows = data[data[key_col] == key]
    if rows.empty:
        return None
    return rows[value_col].iloc[0]


def count_occurrences(x):
    """
    Return a dict mapping each unique value in x to the number of times it appears.
    """
    counts = {}
    for value in x:
        counts[value] = counts.get(value, 0) + 1
    return counts


def list_to_string(x, sep=","):
    """
    Convert a list x to a single string, joining elements with sep.
    """
    return sep.join(str(v) for v in x)


def parse_date(date_str):
    """
    Parse a date string in the form 'YYYY-MM-DD' and return a date object.
    """
    return datetime.strptime(date_str, "%Y-%m-%d").date()
