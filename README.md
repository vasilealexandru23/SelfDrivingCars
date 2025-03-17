### This is an ambitious project making a self-driving vehicle.

### State Estimation Module:
* **input: IMU/GNSS data**
* **output: [pos_x, pos_y, pos_z, speed_x, speed_y, speed_z, q_angle] 10x1 vector**
* **file**: StateEstimation.py

The input from IMU: u=[fk, wk], where fk is 3D specific force in the body frame
and wk is 3D angular rate in the body frame, and u is a 6x1 vector.
This module is responsible for responding to where the car is now. Because we
have some noisy data from IMU and we have some GNSS/GPS data (and also
might have some LiDAR data), we will be using **Error-State Extended Kalman Filter**
to better estimate the car's position, speed and orientation. The estimation
will get continuous, but noisy data from IMU, and that will be used to get the
continuous state of car, and will be corrected by LiDAR/GNSS/GPS data
(at some timestamps).

This module will be used for the motion planning for the vehicle to give more
confidence to the image segmentation module, where we might have some parking
slots avaible or when we are in some intersection and also to plot the trace of
car and it's position in the predefined map.

**TODO**: Calibrate the IMU sensor, and find the proper noise distribution for
best estimation.

### Lane Keeping Module + Object Detection:
* **input: camera image (lores)**
* **output: steering angle, model_output_image(class, box, distance_collision)**
* **file**: LaneKeeping.py

For the **lane estimation** part, we need to detect the lanes of the drivable space.
This is done by selecting a ROI(Region Of Interest) for the left and right lane.
Using Canny-Edge detection, we get the binary image (0 - background, 1 - lane),
and then we save the pairs (x, y) for each 1 class pixel. To get the case where
the lane is a curve, we use cubic spline interpolation. Because our ROI might
get other pixels than the lane(outliers), we use the RANSAC method, where we
select a smaller set of pixels, and get the spline that has the most inliers.

Then, to compute the steering angle, because we can select some point from the
left lane (the same thing applies for selecting the right lane) and because we
know both equations of the splines for left and right lane and also the width of
the road, we can get a line that would coresspond to a segment of the
curvature's radius. The angle of that line will indicate the steering angle.

**TODO**: To better estimate the pixels of the lane (ROI fails very quickly
when the road is not a straight line or when we have outliers from canny-edge
detection), we trained an **Image Segmentation Model(U-Net)** on the [CamVid]
(https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. For now,
we use a U-Net architecture with 10 hidden layers (5 for downsampling and 5 for
upsampling). After 2 hours of training we have 75% accuracy and we need a bigger
architecture and more training time. The CamVid comes with 32 classes, and we
will use the lane-marks, sidewalks, traffic light, road to estimate the drivable
space and the lanemarks pixels. Because it might be highly computational, we
might use another image segmentation, **LaneNet** just for road (this model is
best used for more complex maneuvers, but here is not the case and optimal).

For the **object detection**, we will use the YOLO from ultralytics pre-trained
model(yolov8n) and for distance to collision, we for know use some default
values of the camera's specs. For time to collision, based on our state
estimation module and the mathematical/physical equations, we can estimate
the time to collison.

This task is very important, because it will be used to do safe manuevers, when
the car encounters some static or dynamic elements on the road (other cars or
pedestrians). Based on the boxes found by YOLO, we can estimate at each time how
much the vehicle moved (if it is static or not).

To see if the **object is static**, we will use some feature extraction and feature
matching to see where would the object be if it would be static. If the real
position doesn't match with the estimated postion, that means that the vehicle
is moving and is not static.

If an **object is dynamic**, we can use the **State Estimation Module** to also
find it's future trajectory, first by computing the position, velocity and
orientation. This is very important to give safe maneuvers and to avoid
collision with our vehicle.

Also, the object detection will be responsible to "see" road signs
and change the car's input(speed, steering angle).

### Paperworks + Supplementary Readings
* Read more about taxonomy : Taxonomy and Definitions for Terms Related to Driving Automation Systems : https://www.sae.org/standards/content/j3016_201806/
* Read more about perception task : https://www.cvlibs.net/datasets/kitti/
* Read more about Driving Decision and Actions : https://ieeexplore.ieee.org/abstract/document/7490340 , https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.20255
* Read more about Sensors and Computing Hardware : http://wavelab.uwaterloo.ca/sharedata/ME597/ME597_Lecture_Slides/ME597-4-Measurement.pdf
* Read more about requirements for automated vehicles for highway and rural environments : https://repository.tudelft.nl/record/uuid:2ae44ea2-e5e9-455c-8481-8284f8494e4e 
* Read more about Software Architecture : https://www.semanticscholar.org/paper/DARPA-Urban-Challenge-Technical-Paper-Reinholtz-Alberi/c10acd8c64790f7d040ea6f01d7b26b1d9a442db?p2df
* Read more about Efficient Map Representation : https://ieeexplore.ieee.org/abstract/document/6856487
* Read more about Dynamic Model : https://www.damtp.cam.ac.uk/user/tong/dynamics/clas.pdf and https://publications.lib.chalmers.se/records/fulltext/244369/244369.pdf
* Read more about Longitudinal Vehicle Dynamics : https://link.springer.com/chapter/10.1007/978-1-4614-1433-9_4
* Read more about IMU : https://stanford.edu/class/ee267/lectures/lecture9.pdf
* Read more about Dyamic window approach to collision avoidane : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=580977