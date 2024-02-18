from typing import Set
def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    
    # Convert the set to a list
    fruit_list = list(fruits)
    
    # The range of fruit_list
    if 0 <= fruit_id < len(fruit_list):
        return fruit_list[fruit_id]
    else:
        return "Invalid fruit_id"
