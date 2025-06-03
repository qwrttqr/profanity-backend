def split(rows):
    """
    Split rows based on which fields are changed
    Args:
        rows: list[dict] - rows that should be split

    Returns:
        tuple[list,llist] - profanity and semantic changes array
    """
    profanity_arr = []
    semantic_arr = []
    for item in rows:
        if item['profanity_changed']:
            profanity_arr.append(item)
        if item['semantic_changed']:
            semantic_arr.append(item)

    return profanity_arr, semantic_arr