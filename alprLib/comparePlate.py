from Levenshtein import distance

# custom implementation
# from levenshtein import levenshtein

"""
The goal of this file is to find the best levenshtein match for a given plate.
"""

plates = [
    "CZ233BN",
    "JI548LP",
    "DU602ML",
    "AB123CD",
    "EF456GH",
    "IJ789KL",
    "MN012OP",
    "QR345ST",
    "UV678WX",
    "YZ901AB",
    "CD234EF",
    "GH567IJ",
    "KL890MN",
    "OP123QR",
    "ST456UV",
    "WX789YZ",
    "AB012CD",
    "EF345GH",
    "IJ678KL",
    "MN901OP",
    "QR234ST",
    "UV567WX",
    "YZ890AB",
    "CD123EF",
    "GH456IJ",
    "KL789MN",
    "OP012QR",
    "ST345UV",
    "WX678YZ",
    "AB901CD",
    "EF234GH",
]

def compare_plate(plate: str) -> (str, float):
    best_distance = 1000
    best_match = ""
    for p in plates:
        current_distance = distance(p, plate)
        if current_distance < best_distance:
            best_distance = current_distance
            best_match = p
    return best_match, best_distance

if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) == 2:
        t0 = time.time()
        print(compare_plate(sys.argv[1]))
        print(f"took {time.time() - t0}s")
    else:
        print("Usage: python3 comparePlate.py <plate>")