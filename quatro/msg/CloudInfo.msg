std_msgs/Header header 

int32[] startringindex
int32[] endringindex

float32 startorientation
float32 endorientation
float32 orientationdiff

bool[]    segmentedcloudgroundflag # true - ground point, false - other points
uint32[]  segmentedcloudcolind # point column index in range image
float32[] segmentedcloudrange # point range 
