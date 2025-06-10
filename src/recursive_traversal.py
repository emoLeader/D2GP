import json

def find_chain_info(data, target_objects):
    # Store all hierarchical relationship information
    all_location_information_dic = {}
    all_location_information_list = []
    # Store all encountered objects
    all_objects = set(target_objects)  # Initialize with the target object set
    # Store all related objects
    all_related_objects = set()

    for target_object in target_objects:
        chain_info = []
        
        # Search for target object and its hierarchical relations
        for relation in data:
            if relation['subject'] == target_object and relation['predicate'] in ['isOn', 'isIn']:
                chain_info.append(f"{target_object} {relation['predicate']} {relation['object']}")
                # Add related object
                all_related_objects.add(relation['object'])
                all_objects.add(relation['object'])  # Add to all object set
                # Continue upward search
                next_object = relation['object']
                while True:
                    found = False
                    for r in data:
                        if r['subject'] == next_object and r['predicate'] in ['isOn', 'isIn']:
                            chain_info.append(f"{next_object} {r['predicate']} {r['object']}")
                            # Add related object
                            all_related_objects.add(r['object'])
                            all_objects.add(r['object'])  # Add to all object set
                            next_object = r['object']
                            found = True
                            break
                    if not found:
                        break
        
        # Store result in dictionary
        all_location_information_dic[target_object] = chain_info
        # Store result in list
        all_location_information_list.extend(chain_info)

    return all_location_information_dic, all_location_information_list, all_objects

def get_has_state(data, objects):
    has_state_info = []
    for obj in objects:
        for item in data:
            if item['subject'] == obj and item['predicate'] == 'hasState':
                has_state_info.append(f"{obj} hasState {item['object']}")
    return has_state_info