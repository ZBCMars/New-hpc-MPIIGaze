// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "Regressor.h"
#include "Graphics.h"

using namespace dlib;
using namespace std;
#define RIGHT 4
#define LEFT 5

volatile int pixW = 200, pixH=200;
volatile bool quit=false;
void GraphicsThread() { 
	
	//bool quit = false;
    SDL_Event event;
    //Graphics graphics;
    
    //graphics.init();
   // graphics.setPos(pixW,pixH,false);
    //SDL_Delay(3000);
    //graphics.setPos(700,700,false);
    //SDL_Delay(3000);
   // graphics.setPos(400,400,false);
    while (!quit) {
        SDL_WaitEvent(&event);
        switch (event.type) {
            case SDL_QUIT:
                quit = true;
                break;
        }
    //    graphics.setPos(pixW,pixH,false);
    }
 //   graphics.close(); 
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
     
}
 
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
//cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
void rotationMatrixToEulerAngles(cv::Mat &R, float *headpose, int type)
{
//	cout << "Rotation is:" << cv::Vec3f(180.0/M_PI*atan2(R.at<double>(2,1),R.at<double>(2,2)),
//					  180.0/M_PI*atan2(-R.at<double>(2,0),sqrt(R.at<double>(2,1)*R.at<double>(2,1)+R.at<double>(2,2)*R.at<double>(2,2)) ),
//					  180.0/M_PI*atan2(R.at<double>(1,0),R.at<double>(0,0))					  	
//					 ) << endl;
/*
	assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // If
    float x, y, z;
    if (!singular){
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else{
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    float theta = atan(x/z);
    float phi = asin(-y);
    static int l=0;
    if (l == 1) {
    //cout << "Theta:" << 180.0/M_PI*theta << " and Phi:" << 180.0/M_PI*phi << endl;
    l=0;
	}
	else
		l=1;
    //cout << "Rotation is: " << cv::Vec3f(x, y, z) << endl;
    headpose[0]=theta;
    headpose[1]=phi;
    return;
*/
    //if (type == LEFT) {
    	//afto doulevei ok
    	headpose[0] = -atan2(R.at<double>(0,2),R.at<double>(2,2));//theta
    	headpose[1] =-asin(R.at<double>(1,2));//phi

    	//afto tha prepe na doulevei
    	
    //}
   // else {
    	//edw kanoyme flip kanonika 
   // 	headpose[0] = asin(R.at<double>(1,2));
   //     headpose[1] = -atan2(R.at<double>(0,2),R.at<double>(2,2));
    
        //headpose[0] = -atan2(R.at<double>(0,2),R.at<double>(2,2));
    	//headpose[1] =-asin(R.at<double>(1,2));
    //}

	//return cv::Vec3f(-asin(R.at<double>(1,2)),-atan2(R.at<double>(0,2),R.at<double>(2,2)),1.0);

/*
    assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);   
*/
}
int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../../Downloads/shape_predictor_68_face_landmarks.dat") >> pose_model;
	
		
 

        //define our 3D face model:
        std::vector<cv::Point3d> model3Dpoints;
        model3Dpoints.push_back(cv::Point3d(-45.097f, -0.48377f, 2.397f));// Right eye: Right Corner
        model3Dpoints.push_back(cv::Point3d(-21.313f, 0.48377f, -2.397f));// Right eye: Left Corner
        model3Dpoints.push_back(cv::Point3d(21.313f, 0.48377f, -2.397f));// Left eye: Right Corner
        model3Dpoints.push_back(cv::Point3d(45.097f, -0.48377f, 2.397f));// Left eye: Left Corner
        model3Dpoints.push_back(cv::Point3d(-26.3f, 68.595f, -9.8608e-32f));// Mouth: Right Corner
        model3Dpoints.push_back(cv::Point3d(26.3f, 68.595f, -9.8608e-32f));//Mouth: Left Corner  

        cv::Mat face3d = (cv::Mat_<float>(3,4) << -45.097, -21.313, 21.313, 45.097 , -0.48377,0.48377, 0.48377, -0.48377, 2.397,-2.397,-2.397,2.397);
        face3d.convertTo(face3d, CV_32FC1);
        //face3d = face3d.t();


        // 1c.Calculate xr(x axis) of the head coordinate system
        //cv::Mat xr =(leftmidpoints-rightmidpoints);///cv::norm(leftmidpoints-rightmidpoints);
        //cv::Point3d xr =(leftmidpoints-rightmidpoints)/cv::norm(leftmidpoints-rightmidpoints);

        Regressor regressor;
        regressor.load_model();
        //Create thread for graphics
        std::thread t1(GraphicsThread);
        SDL_Event event;
	    Graphics graphics;
	    graphics.init();
	    graphics.setPos(pixW,pixH,false);

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed() && !quit)
        {
            		
            // Grab a frame
            cv::Mat temp;
            //cv::flip(temp, temp, 1);
            if (!cap.read(temp))
            {
                break;
            }
            //printf("rows are %d\n", temp.rows);//cols=640,rows=460
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);
        

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes2d;

            //to for-loop afto ekteleitai mono an uparxoun faces stin eikona
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = pose_model(cimg,faces[i]);//to pose model den einai function!
                std::vector<cv::Point2d> image_points;


                //we know the (x,y) image landmark positions
                image_points.push_back(cv::Point2d(shape.part(36).x(),shape.part(36).y()));    // Nose tip
                image_points.push_back(cv::Point2d(shape.part(39).x(),shape.part(39).y()));    // Chin
                image_points.push_back(cv::Point2d(shape.part(42).x(),shape.part(42).y()));    // Left eye left corner
                image_points.push_back(cv::Point2d(shape.part(45).x(),shape.part(45).y()));    // Right eye right corner
                image_points.push_back(cv::Point2d(shape.part(48).x(),shape.part(48).y()));    // Left Mouth corner
                image_points.push_back(cv::Point2d(shape.part(54).x(),shape.part(54).y()));    // Right mouth corner
                //tha xreiastoume ta 6 parakatw 2d landmarks(se pixels):
                //n.36: deksia akri deksiou matiou
                //n.39: aristeri akri deksiou matiou
                //n.42: deksia akri aristerou matiou
                //n.45: aristeri akri aristerou matiou
                //n.48: deksia akri stoma
                //n.54: aristeri akri stoma                                
                shapes2d.push_back(pose_model(cimg, faces[i]));//size=68, shape.part(0)         
        		double focal_length = temp.cols;// Approximate focal length.
   				cv::Point2d center = cv::Point2d(temp.cols/2,temp.rows/2);//rows=460, cols=640
    			
                cv::Mat Cr = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
                Cr.convertTo(Cr, CV_32FC1);
    			cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);// Assuming no lens distortion
				cv::Mat rotation_vector; // Rotation in axis-angle form
    			cv::Mat tvec;	
    			cv::solvePnP(model3Dpoints, image_points,Cr, dist_coeffs, rotation_vector, tvec);
                //cout << rotation_vector << endl;
                
                cv::Mat Rr;
                cv::Rodrigues(rotation_vector, Rr);Rr.convertTo(Rr, CV_32FC1);
                //cout << "Rr:"<<Rr << endl;
                tvec.convertTo(tvec, CV_32FC1);
                cv::Mat rotface3d;rotface3d.convertTo(rotface3d, CV_32FC1);

                rotface3d = Rr*face3d;//(3x3)x(3x4)
                for (int o=0;o<4;o++) {
                    rotface3d.col(o) = rotface3d.col(o) + tvec.col(0);   
                }
                cv::Mat reye=0.5*(rotface3d.col(0)
                                             +rotface3d.col(1));
                cv::Mat left_eye_center = 0.5*(rotface3d.col(2)
                                             +rotface3d.col(3));

                //cout << "right eye center:" << right_eye_center << endl;

                float z_scale = 600/cv::norm(reye);
                int fx = 960;int fy = 960;int cx = 30;int cy = 18;
                int NWIDTH = 60;
                int NHEIGHT = 36;
                cv::Mat Cn = (cv::Mat_<float>(3, 3) << fx,0,cx,0,fy,cy,0,0,1);              
                Cn.convertTo(Cn, CV_32FC1);
                cv::Mat S = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z_scale);//cv::magnitude(cameramidpoints));
                S.convertTo(S, CV_32FC1);

                cv::Point3d hRx = (cv::Point3d)Rr.col(0);
                cv::Point3d forward = ((cv::Point3d)reye/cv::norm(reye));
                cv::Point3d down = forward.cross(hRx);
                down = down/cv::norm((cv::Mat)down, cv::NORM_L2, cv::noArray());

                cv::Point3d right = down.cross(forward);
                right=right/cv::norm((cv::Mat)right, cv::NORM_L2, cv::noArray());
                //cout << (cv::Mat)right << endl;
                cv::Mat R = (cv::Mat_<float>(3, 3) << ((cv::Mat)right).at<double>(0),((cv::Mat)down).at<double>(0),((cv::Mat)forward).at<double>(0)
                                                    , ((cv::Mat)right).at<double>(1),((cv::Mat)down).at<double>(1),((cv::Mat)forward).at<double>(1)
                                                    , ((cv::Mat)right).at<double>(2),((cv::Mat)down).at<double>(2),((cv::Mat)forward).at<double>(2));
                R.convertTo(R, CV_32FC1);
                R=R.t();
                //cout << "Matrix R:" << R << endl;
                cv::Mat M = S * R;M.convertTo(M, CV_32FC1);// rotation_matrix.inv();//.t();
                //cout << "MatrixM:" << M << endl;
                cv::Mat W = Cn * M * Cr.inv();
                //cout << "matrix W:" << W << endl;
                cv::Mat output = cv::Mat::zeros(cv::Size(NWIDTH, NHEIGHT),CV_8U/*CV_32FC1*/); 
                cv::warpPerspective(temp,output, W, output.size());
                
                rotation_vector.convertTo(rotation_vector, CV_32FC1);
                //cv::Mat rotvecn = M*rotation_vector;
                //cv::Mat Rn,R_new;
                //cv::Rodrigues(rotvecn,Rn);
                cv::Mat Rn = R*Rr;
                //cv::Mat rvecn;
                //cv::Rodrigues(Rn,rvecn);

                //rvecn = rvecn*(180.0/M_PI);

                //cv::Rodrigues(rvecn,Rn);
                //cout << "rvec is:"<<rvecn << endl;//rvecn=rvecn.row(0);

               //cout << "rvec normalized:"<<rvecn<<endl;
               // << rvecn.at<double>(0,0)//*(180.0/M_PI)
                 //                          <<","<< rvecn.at<double>(0,1)//*(180.0/M_PI)
                  //                          <<","<< rvecn.at<double>(0,2)/**(180.0/M_PI)*/<<endl;
                float theta = asin(Rn.at<float>(1,2));//theta
                //float phi =   atan2(Rn.at<float>(0,2),Rn.at<float>(2,2));//phi
                float phi =   atan(Rn.at<float>(0,2)/Rn.at<float>(2,2));//phi
                //float theta = asin(rvecn.at<float>(1,0));
                //float phi = atan2(rvecn.at<float>(0,0), rvecn.at<float>(2,0));
                //theta = asin(rvecn.at<float>(1,0));
                //phi = atan2(rvecn.at<float>(0,0),rvecn.at<float>(2,0));


                //cv::Rodrigues(Rn,R_new);        
                //cout << "Rn is:" << Rn << endl;
                //cout << "R_new is:" << R_new << endl;

                //float theta = asin(Rn.at<float>(1,0));
                //float phi = atan2(Rn.at<float>(0,0),Rn.at<float>(2,0));
                //0.159155
                //phi = phi - 1.5708;
                //if (phi <0)
                //    phi = -(1.508 +phi);
                //else
                //    phi = 1.508-phi;
                //phi=0.352451;
                //theta=-0.197364;

                //cout << "pose:(" << phi/**(180.0/M_PI)*/<<","<<theta/**(180.0/M_PI)*/<<")"<<endl;
                //cout << "pose1:(" <<theta1* 180.0/M_PI <<","<<phi1* (180.0/M_PI)<<")"<<endl;

                //gaze
                cv::flip(output, output, 0);
                cv::Mat output_orig=output;


                cvtColor( output, output, CV_BGR2GRAY );/// Convert to grayscale
                equalizeHist( output, output);/// Apply Histogram Equalization
                float pose[2]; pose[0] = phi;pose[1]=theta;
                float gaze_n[2];

                regressor.predict(pose,output.data,gaze_n);
                //cout << "***** start array *****" << endl;
                //for (int q=0;q<36;q++) {
                //    for (int r=0;r<60;r++) {
                //        cout << (int)output.at<unsigned char>(q,r) << ",";                    
                //    }
                //    cout << endl;
                //}
                //cout << endl;
                //TODO:Allakse ton Face-Detector wste na mporei na 
                //pianei kai meros tou proswpou

                //cout << "prediction:(" << gaze_n[1]* 180.0/M_PI << "," << gaze_n[0]* 180.0/M_PI << ")" << endl;
                //cout << "tvec:" << tvec << endl;
                //cout << "right_eye_center:" << reye << endl;

                //cv::Mat gazeout = R.inv()*(cv::Mat_<float>(3, 1) << gaze[0], gaze[1], 0);
                //cv::Mat gazeout = R.inv()*(cv::Mat_<float>(3, 1) << gaze[0], gaze[1], 0);   
            	//cout << "final pred:" <<  gazeout* 180.0/M_PI << endl;
  				//ws apostasi mporw na thewrisw to translation_vector me antitheto z-aksona
           		float dx,dy;
				           	      			
           		//tan(gaze[0]) = (dx+x)/(-tvec(2))
           		//tan(gaze[1])= (dy+tvec(1))/(-tvec(2)) 
           	    cv::Mat gaze_r = R.inv()*(cv::Mat_<float>(3, 1) << gaze_n[1], gaze_n[0], 0);
                //cout << "gaze is: " << gaze_r << endl;
                cout << "prediction:(" << gaze_r.at<float>(1,0)* 180.0/M_PI << "," << 
                gaze_r.at<float>(0,0)* 180.0/M_PI << ")" << endl;

                //to dx einai i metatopisi apo to (0,0), diladi aptin kamera
                if (reye.at<float>(0,0) <0 && gaze_r.at<float>(0,0) > 0) { 
                    dx = -reye.at<float>(0,0)+ reye.at<float>(2,0)*tan(gaze_r.at<float>(0,0));
                    pixW=620-dx/0.3647;//*4;//mm se pixels
                    cout <<"case1:" <<pixW<<"dx:"<<dx << endl;
                }
                else if (reye.at<float>(0,0) <0 && gaze_r.at<float>(0,0) < 0) {
                    dx = reye.at<float>(2,0)*tan(gaze_r.at<float>(0,0));
                    pixW = 620+(reye.at<float>(0,0)+dx)/0.3647;///0.3647;//1mm=0.3647px
                    cout <<"case2:"<<pixW<<"dx:"<<dx  << endl;
                } 
                else if (reye.at<float>(0,0)>0 && gaze_r.at<float>(0,0) >0) {
                    dx = (-reye.at<float>(2,0))*tan(gaze_r.at<float>(0,0));
                    pixW=620+(reye.at<float>(0,0)+dx)/0.3647;
                    cout <<"case3:"<<pixW<<"dx:"<<dx  << endl;
                }
                else if (reye.at<float>(0,0)>0 && gaze_r.at<float>(0,0) <0) {
                    dx = reye.at<float>(0,0) - reye.at<float>(2,0)*tan(gaze_r.at<float>(0,0));
                    pixW=620+dx/0.3647;
                    cout <<"case4:"<<pixW<<"dx:"<<dx  << endl;
                }
                //pixW=620;
           		
                //dx = -tan(gaze_r.at<float>(1,0))*reye.at<float>(2,0) -reye.at<float>(0,0);//in mm's

      			//dy = -tan(gaze_r.at<float>(0,0))*reye.at<float>(2,0) - reye.at<float>(1,0);//in mm's
      			dy=50;
                //cout << "translation_vector:" << translation_vector << endl;
                //cout << "a1:" << gaze_r.at<float>(0,0) << endl;
                //cout << "a3:" << translation_vector.at<float>(1 ,0) << endl;//in mm's
                //cout << "a2:" << translation_vector.at<float>(2,0) << endl;
                //cout << "a3:" << translation_vector.at<float>(0,0) << endl;//in mm's
                //cout  << " dx is " << dx<<", dy is " << dy<< endl;
      			//cout << "theta:("<< gaze_r.at<float>(1,0)* 180.0/M_PI<<",phi:" << gaze_r.at<float>(0,0)* 180.0/M_PI<<")"<< endl;
                //cout  << "dist:(" << dx <<"," << dy<<")"<< endl;
                //-1.21378 gaze_r.at<float>(0,0)
                //-584.579 translation_vector.at<float>(2,0)
                //-61.781 translation_vector.at<float>(0,0)
                //-44.2272 translation_vector.at<float>(1,0)

                //dist:(-1505.44,1078.35)
                //dx = -tan(-1.21378)*(-584.579) - (-61.781)


           		dy = 4 * dy;
           		//dx =  4 * dx + 340/2;
           		pixH = dy;//set
           		//pixW = dx;

           		//othoni diastaseis:W=1240px kai 340cm kai H = 780px kai 190cm
                //1240 = 3400mm
                //???  = 1mm=0.3647px
                //???=0.3647

           		//cout  << "pixel:(" << pixW <<"," << pixH<<")"<< endl;
                graphics.setPos(pixW,pixH,false);


                cv_image<bgr_pixel> cimg2(output_orig);
                win.clear_overlay();
                win.set_image(cimg2);
                win.add_overlay(render_face_detections(shapes2d)); 
            }
            

            //Now let's view our face poses on the screen.
            
            //win.clear_overlay();
            //win.set_image(cimg);
            //win.add_overlay(render_face_detections(shapes2d)); 
	    		    
        }
        regressor.close();
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

	

/*
    	vector<Point3d> nose_end_point3D;//Project a 3D point (0,0,1000.0) onto the image plane.
    	vector<Point2d> nose_end_point2D;
    	nose_end_point3D.push_back(Point3d(0,0,1000.0));
     
    	projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
*/



//dlib::array<array2d<rgb_pixel> > face_chips;
//extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
//win.set_image(tile_images(face_chips));	
        
