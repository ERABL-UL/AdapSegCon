# This file is covered by the LICENSE file in the root of this project.
labels = {
  0 : "unclassified",
  1 : "ground",
  2: "building",
  3: "pole - road sign - traffic light",
  4: "bollard - small pole",
  5: "trash can",
  6: "barrier",
  7: "pedestrian",
  8: "car",
  9: "natural - vegetation"
}
# classes that are indistinguishable from single scan or inconsistent in
# ground truth are mapped to their closest equivalent
learning_map = {
  0: 0,      # "unclassified"
  1: 1,      # "ground"
  2: 2,      # "building"
  3: 3,      # "pole - road sign - traffic light"
  4: 4,      # "bollard - small pole"
  5: 5,      # "trash can" 
  6: 6,      # "barrier"    
  7: 7,      # "pedestrian"   
  8: 8,      # "car"       
  9: 9       # "natural - vegetation"      
}
learning_map_inv = { # inverse of previous map
  0: 0,      # "unclassified"  
  1: 1,      # "ground"
  2: 2,      # "building"
  3: 3,      # "pole - road sign - traffic light" 
  4: 4,      # "bollard - small pole"      
  5: 5,      # "trash can"    
  6: 6,      # "barrier"      
  7: 7,      # "pedestrian"  
  8: 8,      # "car"
  9: 9       # "natural - vegetation"
}
learning_ignore = { # Ignore classes
  0: True,       # "unclassified"  
  1: False,      # "ground"
  2: False,      # "building"
  3: False,      # "pole - road sign - traffic light" 
  4: False,      # "bollard - small pole"      
  5: False,      # "trash can"    
  6: False,      # "barrier"      
  7: False,      # "pedestrian"  
  8: False,      # "car"
  9: False       # "natural - vegetation"
}

poss_map = {
  0: 0,       # "unclassified"  
  1: 1,      # "ground"
  2: 2,      # "building"
  3: 3,      # "pole - road sign - traffic light" 
  4: 4,      # "bollard - small pole"      
  5: 5,      # "trash can"    
  6: 6,      # "barrier"      
  7: 7,      # "pedestrian"  
  8: 8,      # "car"
  9: 9       # "natural - vegetation"
}


labels_poss = {
  0: 'unclassified',
  1: 'ground',
  2: 'building',
  3: 'pole - road sign - traffic light',
  4: 'bollard - small pole',
  5: 'trash can',
  6: 'barrier',   
  7: 'pedestrian',
  8: 'car',
  9: 'natural - vegetation'
}
