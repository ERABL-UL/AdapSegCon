# This file is covered by the LICENSE file in the root of this project.
labels = {
  0 : "unlabeled",
  1 : "road",
  2: "sidewalk",
  3: "building",
  4: "wall",
  5: "fence",
  6: "pole",
  7: "traffic light",
  8: "traffic sign",
  9: "vegetation",
  10: "terrain",
  11: "person",
  12: "car",
  13: "truck",
  14: "motorcycle",
  15: "bicycle",
  16: "tunnel",
  17: "pole",
  18: "polegroup",
  19: "traffic light",
  20: "traffic sign",
  21: "vegetation",
  22: "terrain",
  23: "sky",
  24: "person",
  25: "rider",
  26: "car",
  27: "truck",
  28: "bus",
  29: "caravan",
  30: "trailer",
  31: "train",
  32: "motorcycle",
  33: "bicycle",
  34: "garage",
  35: "gate",
  36: "stop",
  37: "smallpole",
  38: "lamp",
  39: "trash bin",
  40: "vending machine",
  41: "box",
  42: "unknown construction",
  43: "unknown vehicle",
  44: "unknown object",
  45: "license plate"
}
# classes that are indistinguishable from single scan or inconsistent in
# ground truth are mapped to their closest equivalent
learning_map = {
  0: 0,      # "unlabeled"               mapped to "unlabeled" --------------------------mapped
  1: 0,      # "ego vehicle"             mapped to "unlabeled" --------------------------mapped
  2: 0,      # "rectification border"    mapped to "unlabeled" --------------------------mapped
  3: 0,      # "out of roi"              mapped to "unlabeled" --------------------------mapped
  4: 0,      # "static"                  mapped to "unlabeled" --------------------------mapped
  5: 0,      # "dynamic"                 mapped to "unlabeled" --------------------------mapped
  6: 0,      # "ground"                  mapped to "unlabeled" --------------------------mapped
  7: 1,      # "road"                    mapped to "road" -------------------------------mapped
  8: 2,      # "sidewalk"                mapped to "sidewalk" ---------------------------mapped
  9: 0,      # "parking"                 mapped to "unlabeled" --------------------------mapped
  10: 0,     # "rail track"              mapped to "unlabeled" --------------------------mapped
  11: 3,     # "building"                mapped to "building" ---------------------------mapped
  12: 4,     # "wall"                    mapped to "wall" -------------------------------mapped
  13: 5,     # "fence"                   mapped to "fence" ------------------------------mapped
  14: 0,     # "guard rail"              mapped to "unlabeled" --------------------------mapped
  15: 0,     # "bridge"                  mapped to "unlabeled" --------------------------mapped
  16: 0,     # "tunnel"                  mapped to "unlabeled" --------------------------mapped
  17: 6,     # "pole"                    mapped to "pole" -------------------------------mapped
  18: 0,     # "polegroup"               mapped to "unlabeled" --------------------------mapped
  19: 7,     # "traffic light"           mapped to "traffic light" ----------------------mapped
  20: 8,     # "traffic sign"            mapped to "traffic sign" -----------------------mapped
  21: 9,     # "vegetation"              mapped to "vegetation" -------------------------mapped
  22: 10,    # "terrain"                 mapped to "terrain" ----------------------------mapped
  23: 0,     # "sky"                     mapped to "unlabeled" --------------------------mapped
  24: 11,    # "person"                  mapped to "person" -----------------------------mapped
  25: 0,     # "rider"                   mapped to "unlabeled" --------------------------mapped
  26: 12,    # "car"                     mapped to "car" --------------------------------mapped
  27: 13,    # "truck"                   mapped to "truck" ------------------------------mapped
  28: 0,     # "bus"                     mapped to "unlabeled" --------------------------mapped
  29: 0,     # "caravan"                 mapped to "unlabeled" --------------------------mapped
  30: 0,     # "trailer"                 mapped to "unlabeled" --------------------------mapped
  31: 0,     # "train"                   mapped to "unlabeled" --------------------------mapped
  32: 14,    # "motorcycle"              mapped to "motorcycle" -------------------------mapped
  33: 15,    # "bicycle"                 mapped to "bicycle" ----------------------------mapped
  34: 3,     # "garage"                  mapped to "building" ---------------------------mapped
  35: 5,     # "gate"                    mapped to "fence" ------------------------------mapped
  36: 0,     # "stop"                    mapped to "unlabeled"---------------------------mapped
  37: 6,     # "smallpole"               mapped to "pole"--------------------------------mapped
  38: 0,     # "lamp"                    mapped to "unlabeled"---------------------------mapped
  39: 0,     # "trash bin"               mapped to "unlabeled"---------------------------mapped
  40: 0,     # "vending machine"         mapped to "unlabeled"---------------------------mapped
  41: 0,     # "box"                     mapped to "unlabeled"---------------------------mapped
  42: 0,     # "unknown construction"    mapped to "unlabeled"---------------------------mapped
  43: 0,     # "unknown vehicle"         mapped to "unlabeled"---------------------------mapped
  44: 0,     # "unknown object"          mapped to "unlabeled"---------------------------mapped
  45: 0      # "license plate"           mapped to "unlabeled" --------------------------mapped
}
learning_map_inv = { # inverse of previous map
  0: 0,      # "unlabeled"               mapped to "unlabeled" --------------------------mapped
  0: 1,      # "ego vehicle"             mapped to "unlabeled" --------------------------mapped
  0: 2,      # "rectification border"    mapped to "unlabeled" --------------------------mapped
  0: 3,      # "out of roi"              mapped to "unlabeled" --------------------------mapped
  0: 4,      # "static"                  mapped to "unlabeled" --------------------------mapped
  0: 5,      # "dynamic"                 mapped to "unlabeled" --------------------------mapped
  0: 6,      # "ground"                  mapped to "unlabeled" --------------------------mapped
  1: 7,      # "road"                    mapped to "road" -------------------------------mapped
  2: 8,      # "sidewalk"                mapped to "sidewalk" ---------------------------mapped
  0: 9,      # "parking"                 mapped to "unlabeled" --------------------------mapped
  0: 10,     # "rail track"              mapped to "unlabeled" --------------------------mapped
  3: 11,     # "building"                mapped to "building" ---------------------------mapped
  4: 12,     # "wall"                    mapped to "wall" -------------------------------mapped
  5: 13,     # "fence"                   mapped to "fence" ------------------------------mapped
  0: 14,     # "guard rail"              mapped to "unlabeled" --------------------------mapped
  0: 15,     # "bridge"                  mapped to "unlabeled" --------------------------mapped
  0: 16,     # "tunnel"                  mapped to "unlabeled" --------------------------mapped
  6: 17,     # "pole"                    mapped to "pole" -------------------------------mapped
  0: 18,     # "polegroup"               mapped to "unlabeled" --------------------------mapped
  7: 19,     # "traffic light"           mapped to "traffic light" ----------------------mapped
  8: 20,     # "traffic sign"            mapped to "traffic sign" -----------------------mapped
  9: 21,     # "vegetation"              mapped to "vegetation" -------------------------mapped
  10: 22,    # "terrain"                 mapped to "terrain" ----------------------------mapped
  0: 23,     # "sky"                     mapped to "unlabeled" --------------------------mapped
  11: 24,    # "person"                  mapped to "person" -----------------------------mapped
  0: 25,     # "rider"                   mapped to "unlabeled" --------------------------mapped
  12: 26,    # "car"                     mapped to "car" --------------------------------mapped
  13: 27,    # "truck"                   mapped to "truck" ------------------------------mapped
  0: 28,     # "bus"                     mapped to "unlabeled" --------------------------mapped
  0: 29,     # "caravan"                 mapped to "unlabeled" --------------------------mapped
  0: 30,     # "trailer"                 mapped to "unlabeled" --------------------------mapped
  0: 31,     # "train"                   mapped to "unlabeled" --------------------------mapped
  14: 32,    # "motorcycle"              mapped to "motorcycle" -------------------------mapped
  15: 33,    # "bicycle"                 mapped to "bicycle" ----------------------------mapped
  3: 34,     # "garage"                  mapped to "building" ---------------------------mapped
  5: 35,     # "gate"                    mapped to "fence" ------------------------------mapped
  0: 36,     # "stop"                    mapped to "unlabeled"---------------------------mapped
  6: 37,     # "smallpole"               mapped to "pole"--------------------------------mapped
  0: 38,     # "lamp"                    mapped to "unlabeled"---------------------------mapped
  0: 39,     # "trash bin"               mapped to "unlabeled"---------------------------mapped
  0: 40,     # "vending machine"         mapped to "unlabeled"---------------------------mapped
  0: 41,     # "box"                     mapped to "unlabeled"---------------------------mapped
  0: 42,     # "unknown construction"    mapped to "unlabeled"---------------------------mapped
  0: 43,     # "unknown vehicle"         mapped to "unlabeled"---------------------------mapped
  0: 44,     # "unknown object"          mapped to "unlabeled"---------------------------mapped
  0: 45      # "license plate"           mapped to "unlabeled" --------------------------mapped
}
learning_ignore = { # Ignore classes
  0: True,       # "unlabeled"
  1: False,      # "road"
  2: False,      # "sidewalk"
  3: False,      # "building"
  4: False,      # "wall"
  5: False,      # "fence"
  6: False,      # "pole"
  7: False,      # "traffic light"
  8: False,      # "traffic sign"
  9: False,      # "vegetation"
  10: False,      # "terrain"
  11: False,      # "person"
  12: False,      # "car"
  13: False,      # "truck"
  14: False,      # "motorcycle"
  15: False      # "bicycle"
}


poss_map = {
  0: 0,       # "unlabeled"
  1: 1,      # "road"
  2: 2,      # "sidewalk"
  3: 3,      # "building"
  4: 4,      # "wall"
  5: 5,      # "fence"
  6: 6,      # "pole"
  7: 7,      # "traffic light"
  8: 8,      # "traffic sign"
  9: 9,      # "vegetation"
  10: 10,      # "terrain"
  11: 11,      # "person"
  12: 12,      # "car"
  13: 13,      # "truck"
  14: 14,      # "motorcycle"
  15: 15      # "bicycle"
}


labels_poss = {
  0 : "unlabeled",
  1 : "road",
  2: "sidewalk",
  3: "building",
  4: "wall",
  5: "fence",
  6: "pole",
  7: "traffic light",
  8: "traffic sign",
  9: "vegetation",
  10: "terrain",
  11: "person",
  12: "car",
  13: "truck",
  14: "motorcycle",
  15: "bicycle"
}
