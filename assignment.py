dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}

dict1.update(dict2)
print("Merged dictionary:", dict1)

dict3 = {"x": 10, "y": 20}
dict4 = {"z": 30}

merged = {**dict3, **dict4}
print("Merged dictionary:", merged)