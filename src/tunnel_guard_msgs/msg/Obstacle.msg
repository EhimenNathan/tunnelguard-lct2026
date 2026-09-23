# One confirmed foreign object inside (or touching) the train's dynamic envelope.
uint8 ZONE_NEAR_GAUGE = 1     # within the safety margin around the envelope
uint8 ZONE_IN_GAUGE   = 2     # intersects the train envelope

uint32 id                      # persistent track id
uint8 zone
float32 distance               # along-track distance from the lidar to the nearest point [m]
float32 range                  # euclidean distance to the nearest point [m]
geometry_msgs/Point position   # cluster centroid in the lidar frame (header.frame_id)
geometry_msgs/Vector3 size     # extent along track / lateral / vertical [m]
float32 lateral_offset         # signed offset of the centroid from the track centreline [m], +left
float32 height_above_rail      # highest point above the rail plane [m]
uint32 num_points
float32 confidence             # 0..1, grows with persistence and point support
float32 closing_speed          # rate of distance decrease [m/s], >0 when approaching
float32 time_to_collision      # [s], +inf when not closing
uint32 age_frames              # consecutive frames this object has been observed
