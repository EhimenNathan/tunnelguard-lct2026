# Per-frame decision for the train control / decision-making subsystem.
uint8 LEVEL_CLEAR   = 0        # path verified clear up to clear_distance
uint8 LEVEL_CAUTION = 1        # object near the envelope, or low-confidence object in it
uint8 LEVEL_STOP    = 2        # confirmed object in the envelope -> brake

std_msgs/Header header         # stamp = stamp of the source point cloud
uint8 level
bool obstacle_detected
float32 nearest_distance       # along-track distance to the nearest in-gauge obstacle [m], NaN if none
float32 nearest_time_to_collision
float32 clear_distance         # distance up to which the envelope was actually observed empty [m]
float32 track_curvature        # estimated curvature of the track ahead [1/m], +left
float32 processing_time_ms     # cloud reception -> decision
Obstacle[] obstacles
