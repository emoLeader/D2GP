def fix_brackets(response_str):

    left_brackets_count = response_str.count('[')
    right_brackets_count = response_str.count(']')
    
    if left_brackets_count < 3:
        response_str = '[' * (3 - left_brackets_count) + response_str
    
    if right_brackets_count < 3:
        response_str = response_str + ']' * (3 - right_brackets_count)
    
    return response_str