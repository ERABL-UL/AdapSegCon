# This file is covered by the LICENSE file in the root of this project.
labels = {
  0 : "Unclassified",
  1 : "Ground",
  2: "Road_markings",
  3: "Natural",
  4: "Building",
  5: "Utility_line",
  6: "Pole",
  7: "Car",
  8: "Fence"
}
# classes that are indistinguishable from single scan or inconsistent in
# ground truth are mapped to their closest equivalent
learning_map = {
  0: 0,      # "Unclassified"
  1: 1,      # "Ground"
  2: 2,      # "Road_markings"
  3: 3,      # "Natural"
  4: 4,      # "Building"
  5: 5,      # "Utility_line" 
  6: 6,      # "Pole"    
  7: 7,      # "Car"   
  8: 8       # "Fence"        
}
learning_map_inv = { # inverse of previous map
  0: 0,      # "Unclassified"
  1: 1,      # "Ground"
  2: 2,      # "Road_markings"
  3: 3,      # "Natural"
  4: 4,      # "Building"
  5: 5,      # "Utility_line" 
  6: 6,      # "Pole"    
  7: 7,      # "Car"   
  8: 8       # "Fence"  
}
learning_ignore = { # Ignore classes
  0: True,      # "Unclassified"
  1: False,      # "Ground"
  2: False,      # "Road_markings"
  3: False,      # "Natural"
  4: False,      # "Building"
  5: False,      # "Utility_line" 
  6: False,      # "Pole"    
  7: False,      # "Car"   
  8: False       # "Fence"  
}


poss_map = {
  0: 0,      # "Unclassified"
  1: 1,      # "Ground"
  2: 2,      # "Road_markings"
  3: 3,      # "Natural"
  4: 4,      # "Building"
  5: 5,      # "Utility_line" 
  6: 6,      # "Pole"    
  7: 7,      # "Car"   
  8: 8       # "Fence"        
}


labels_poss = {
  0: 'Unclassified',
  1: 'Ground',
  2: 'Road_markings',
  3: 'Natural',
  4: 'Building',
  5: 'Utility_line',
  6: 'Pole',
  7: 'Car', 
  8: 'Fence'
}
