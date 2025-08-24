from lgn.dataset import AutoTransformer


def get_parts(Dataset: AutoTransformer, input_ids: list[int]):
    parts: list[list[int]] = []
    start = 0
    for step in Dataset.get_attribute_ranges():
        parts.append(input_ids[start : start + step])
        start += step
    return parts
