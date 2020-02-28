clear;
clc;

addpath('mexopencv-2.4/'); % we need to use some function from OpenCV

% load the face model
faceModel = load('../MPIIGaze/Data/6 points-based face model.mat');
faceModel = faceModel.model;

% load the image, annotation and camera parameters.
img = imread('../MPIIGaze/Data/Original/p00/day01/0001.jpg');
annotation = load('../MPIIGaze/Data/Original/p00/day01/annotation.txt');
cameraCalib = load('../MPIIGaze/Data/Original/p00/Calibration/Camera.mat');


% get head pose
headpose_hr = annotation(1, 30:32);
headpose_ht = annotation(1, 33:35);
hR = rodrigues(headpose_hr); 
Fc = hR* faceModel; % rotate the face model, which is calcluated from facial landmakr detection
Fc = bsxfun(@plus, Fc, headpose_ht);

% get the eye center in the original camera cooridnate system.
right_eye_cetter = 0.5*(Fc(:,1)+Fc(:,2));
left_eye_center = 0.5*(Fc(:,3)+Fc(:,4));

% get the gaze target
gaze_target = annotation(1, 27:29);
gaze_target = gaze_target;

% set the size of normalized eye image
eye_image_width  = 60;
eye_image_height = 36;

% normalization for the right eye, you can do it for left eye by replacing
% "right_eye_cetter" to "left_eye_center"
[eye_img, headpose, gaze] = normalizeImg(img, right_eye_cetter, hR, gaze_target, [eye_image_width, eye_image_height], cameraCalib.cameraMatrix);
imshow(eye_img);

% convert the gaze direction in the camera cooridnate system to the angle
% in the polar coordinate system
gaze_theta = asin((-1)*gaze(2)); % vertical gaze angle
gaze_phi = atan2((-1)*gaze(1), (-1)*gaze(3)); % horizontal gaze angle

% save as above, conver head pose to the polar coordinate system
M = rodrigues(headpose);
Zv = M(:,3);
headpose_theta = asin(Zv(2)); % vertical head pose angle
headpose_phi = atan2(Zv(1), Zv(3)); % horizontal head pose angle
