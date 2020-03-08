/*
 *
 *
 * This benchmark file can run any SLAM algorithm compatible with the SLAMBench API
 * We recommend you to generate a library which is compatible with this application.
 * 
 * The interface works that way :
 *   - First benchmark.cpp will call an initialisation function of the SLAM algorithm. (void sb_init())
 *     This function provides the compatible interface with the SLAM algorithm.
 *   - Second benchmark.cpp load an interface (the textual interface in our case)
 *   - Then for every frame, the benchmark.cpp will call sb_process()
 *
 *
 */
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"


#include <timings.h>
#include <SLAMBenchAPI.h>

#include <io/sensor/Sensor.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/DepthSensor.h>
#include <io/SLAMFrame.h>




std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;


int mode=0;

bool firstRosSpin=false;

using namespace dso;

// ===========================================================
// Default Parameters
// ===========================================================

static const float
      default_confidence = 10.0f,
      default_depth = 3.0f,
      default_icp = 10.0f,
      default_icpErrThresh = 5e-05,
      default_covThresh = 1e-05,
      default_photoThresh = 115,
      default_fernThresh = 0.3095f;
static const int
    default_timeDelta = 200,
    default_icpCountThresh = 35000;

static const int
    default_textureDim = 1024,
    default_nodeTextureDim = 8192;

static const bool
     default_openLoop = false,
     default_reloc = false,
     default_fastOdom = false,
     default_so3 = true,
     default_frameToFrameRGB = false;



// ===========================================================
// Algorithm parameters
// ===========================================================

  float confidence,
          depth,
          icp,
          icpErrThresh,
          covThresh,
          photoThresh,
          fernThresh;

    int  icpCountThresh,
         timeDelta,
         textureDim,
         nodeTextureDim;

    bool openLoop,
         reloc,
         fastOdom,
         so3,
         frameToFrameRGB;

	std::string shader_dir;




// ===========================================================
// ElasticFusion
// ===========================================================


static sb_uchar3* inputRGB;
static sb_uchar4* imageTex;
static unsigned short int * inputDepth;

static sb_uint2 inputSize;


// ===========================================================
// SLAMBench Sensors
// ===========================================================

static slambench::io::DepthSensor *depth_sensor;
static slambench::io::CameraSensor *rgb_sensor;


// ===========================================================
// SLAMBench Outputs
// ===========================================================

slambench::outputs::Output *pose_output;
slambench::outputs::Output *pointcloud_output;

slambench::outputs::Output *rgb_frame_output;
slambench::outputs::Output *render_frame_output;



 bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings)  {

     slam_settings->addParameter(TypedParameter<float>("c", "confidence",     "Confidence",      &confidence, &default_confidence));
     slam_settings->addParameter(TypedParameter<float>("d", "depth",          "Depth",           &depth,      &default_depth));
     slam_settings->addParameter(TypedParameter<float>("", "icp",            "ICP",             &icp, &default_icp));
     slam_settings->addParameter(TypedParameter<float>("ie", "icpErrThresh",   "ICPErrThresh",    &icpErrThresh, &default_icpErrThresh));
     slam_settings->addParameter(TypedParameter<float>("cv", "covThresh",      "CovThresh",       &covThresh, &default_covThresh));
     slam_settings->addParameter(TypedParameter<float>("pt", "photoThresh",    "PhotoThresh",     &photoThresh, &default_photoThresh));
     slam_settings->addParameter(TypedParameter<float>("ft", "fernThresh",     "FernThresh",      &fernThresh, &default_fernThresh));
     slam_settings->addParameter(TypedParameter<int>  ("ic", "icpCountThresh", "ICPCountThresh",  &icpCountThresh, &default_icpCountThresh));
     slam_settings->addParameter(TypedParameter<int>  ("t", "timeDelta",      "TimeDelta",       &timeDelta,      &default_timeDelta));
     slam_settings->addParameter(TypedParameter<int>  ("td", "textureDim",      "textureDim",       &textureDim,      &default_textureDim));
     slam_settings->addParameter(TypedParameter<int>  ("ntd", "nodeTextureDim",      "nodeTextureDim",       &nodeTextureDim,      &default_nodeTextureDim));
     slam_settings->addParameter(TypedParameter<bool>("ol", "openLoop",        "OpenLoop",        &openLoop       , &default_openLoop       ));
     slam_settings->addParameter(TypedParameter<bool>("rl", "reloc",           "Reloc",           &reloc          , &default_reloc          ));
     slam_settings->addParameter(TypedParameter<bool>("fod", "fastOdom",        "FastOdom",        &fastOdom       , &default_fastOdom       ));
     slam_settings->addParameter(TypedParameter<bool>("nso", "so3",             "So3",             &so3            , &default_so3            ));
     slam_settings->addParameter(TypedParameter<bool>("ftf", "frameToFrameRGB", "FrameToFrameRGB", &frameToFrameRGB, &default_frameToFrameRGB));
	 
	 slam_settings->addParameter(TypedParameter<std::string>("sh", "shader-dir", "Directory containing shaders", &shader_dir, &default_shader_dir));
     return true;
 }


void settingsDefault(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");
    if(preset == 0 || preset == 1)
    {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n", preset==0 ? "no " : "1x");

        playbackSpeed = (preset==0 ? 0 : 1);
        preload = preset==1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations=6;
        setting_minOptIterations=1;

        setting_logStuff = false;
    }

    if(preset == 2 || preset == 3)
    {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

        playbackSpeed = (preset==2 ? 0 : 5);
        preload = preset==3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations=4;
        setting_minOptIterations=1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}






void parseArgument(char* arg)
{
    int option;
    float foption;
    char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if(1==sscanf(arg,"preset=%d",&option))
    {
        settingsDefault(option);
        return;
    }


    if(1==sscanf(arg,"rec=%d",&option))
    {
        if(option==0)
        {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }



    if(1==sscanf(arg,"noros=%d",&option))
    {
        if(option==1)
        {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if(1==sscanf(arg,"nolog=%d",&option))
    {
        if(option==1)
        {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if(1==sscanf(arg,"reverse=%d",&option))
    {
        if(option==1)
        {
            reverse = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if(1==sscanf(arg,"nogui=%d",&option))
    {
        if(option==1)
        {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if(1==sscanf(arg,"nomt=%d",&option))
    {
        if(option==1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if(1==sscanf(arg,"prefetch=%d",&option))
    {
        if(option==1)
        {
            prefetch = true;
            printf("PREFETCH!\n");
        }
        return;
    }
    if(1==sscanf(arg,"start=%d",&option))
    {
        start = option;
        printf("START AT %d!\n",start);
        return;
    }
    if(1==sscanf(arg,"end=%d",&option))
    {
        end = option;
        printf("END AT %d!\n",start);
        return;
    }

    if(1==sscanf(arg,"files=%s",buf))
    {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if(1==sscanf(arg,"calib=%s",buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if(1==sscanf(arg,"vignette=%s",buf))
    {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if(1==sscanf(arg,"gamma=%s",buf))
    {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if(1==sscanf(arg,"rescale=%f",&foption))
    {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if(1==sscanf(arg,"speed=%f",&foption))
    {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if(1==sscanf(arg,"save=%d",&option))
    {
        if(option==1)
        {
            debugSaveImages = true;
            if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if(1==sscanf(arg,"mode=%d",&option))
    {

        mode = option;
        if(option==0)
        {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if(option==1)
        {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if(option==2)
        {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}


bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings) {


     /**
      * Retrieve RGB and Depth sensors,
      *  - check input_size are the same
      *  - check camera are the same
      *  - get input_file
      */


	slambench::io::CameraSensorFinder sensor_finder;
	rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
	depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});
	 
    if ((rgb_sensor == nullptr) || (depth_sensor == nullptr)) {
        std::cerr << "Invalid sensors found, RGB or Depth not found." << std::endl;
        return false;
    }

	if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
		std::cerr << "RGB data is in wrong format" << std::endl;
		return false;
	}
	if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
		std::cerr << "Depth data is in wrong format" << std::endl;
		return false;
	}
	if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
		std::cerr << "RGB data is in wrong format pixel" << std::endl;
		return false;
	}
	if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
		std::cerr << "Depth data is in wrong pixel format" << std::endl;
		return false;
	}

	assert(depth_sensor->Width == rgb_sensor->Width);
	assert(depth_sensor->Height == rgb_sensor->Height);
	//assert(depth_sensor->Intrinsics == rgb_sensor->Intrinsics);

     inputSize = make_sb_uint2(rgb_sensor->Width, rgb_sensor->Height);

    FullSystem* fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed==0);







    IOWrap::PangolinDSOViewer* viewer = 0;
    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());



//
//    float4 camera =  make_float4(
//			rgb_sensor->Intrinsics[0],
//			rgb_sensor->Intrinsics[1],
//			rgb_sensor->Intrinsics[2],
//			rgb_sensor->Intrinsics[3]);
//
//     camera.x = camera.x * rgb_sensor->Width;
//     camera.y = camera.y * rgb_sensor->Height;
//     camera.z = camera.z * rgb_sensor->Width;
//     camera.w = camera.w * rgb_sensor->Height;
//
//     // fx, fy, cx, cy
//     std::cerr << "Intrisics are fx:" << camera.x << " fy:" << camera.y << " cx:" << camera.z << " cy:" << camera.w << std::endl;
//     Intrinsics::getInstance(camera.x, camera.y, camera.z, camera.w);
//
//     std::cerr << "OpenGL setup..." << std::endl;
//
//     setup_opengl_context();

     std::cerr << "OpenGL setup is done." << std::endl;


     imageTex = (sb_uchar4*) malloc(
                   sizeof(sb_uchar4) * inputSize.x * inputSize.y);

     inputRGB = (sb_uchar3*) malloc(
                   sizeof(sb_uchar3) * inputSize.x * inputSize.y);
     inputDepth = (uint16_t*) malloc(
                    sizeof(uint16_t) * inputSize.x * inputSize.y);




     pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
     slam_settings->GetOutputManager().RegisterOutput(pose_output);

     pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_COLOUREDPOINTCLOUD, true);
     pointcloud_output->SetKeepOnlyMostRecent(true);
     slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);

     rgb_frame_output = new slambench::outputs::Output("RGB Frame", slambench::values::VT_FRAME);
     rgb_frame_output->SetKeepOnlyMostRecent(true);
     slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);

     render_frame_output = new slambench::outputs::Output("Rendered frame", slambench::values::VT_FRAME);
     render_frame_output->SetKeepOnlyMostRecent(true);
     slam_settings->GetOutputManager().RegisterOutput(render_frame_output);


     return true;
 }


 bool depth_ready = false;
 bool rgb_ready   = false;

bool sb_update_frame (SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s) {

	if (depth_ready and rgb_ready) {
		depth_ready = false;
		rgb_ready   = false;
	}

	assert(s != nullptr);
	
	char *target = nullptr;
//	fullSystem->addActiveFrame(img, i);


    if(s->FrameSensor == depth_sensor) {
		target = (char*)inputDepth;
		depth_ready = true;
	} else if(s->FrameSensor == rgb_sensor) {
		target = (char*)inputRGB;
		rgb_ready = true;
	}
	
	if(target != nullptr) {
		memcpy(target, s->GetData(), s->GetSize());
		s->FreeData();
	}
	
	return depth_ready and rgb_ready;
}

 bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {
     static int frame = 0;
     frame++;
     return true;
 }

 
 bool sb_get_tracked  (bool* tracking)  {
    return true;
}

 bool sb_clean_slam_system() {
     delete inputRGB;
     delete inputDepth;
     return true;
 }



 bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *ts_p) {
		slambench::TimeStamp ts = *ts_p;

		if(pose_output->IsActive()) {
			// Get the current pose as an eigen matrix
			Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();

			std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
			pose_output->AddPoint(ts, new slambench::values::PoseValue(mat));
		}

		if(pointcloud_output->IsActive()) {

		}

		if(rgb_frame_output->IsActive()) {
			std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
			rgb_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, inputRGB));
		}


		if(render_frame_output->IsActive()) {
//		    eFusion->getIndexMap().imageTex()->texture->Download(imageTex, GL_RGBA, GL_UNSIGNED_BYTE);
//			std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
//			render_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGBA_IIII_8888, imageTex));
		}

		return true;

     
     
     return true;
 }

