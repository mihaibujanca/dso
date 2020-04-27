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

#include <IOWrapper/Pangolin/SLAMBenchOutputWrapper.h>
#include <SLAMBenchAPI.h>
#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>

#include <opencv2/core/mat.hpp>

#include "FullSystem/FullSystem.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/DatasetReader.h"
#include "util/NumType.h"
#include "util/Undistort.h"
#include "util/globalCalib.h"
#include "util/settings.h"

std::string vignetteFile, gammaCalibFile, calibFile;
std::string default_vignette, default_gammaCalib, default_calibFile;

int mode, default_mode=1;
dso::FullSystem* fullSystem;
dso::ImageAndExposure* img;
dso::Undistort* undistorter;
IOWrap::SLAMBenchOutputWrapper * wrapper;

static slambench::io::CameraSensor *grey_sensor = nullptr;
static slambench::outputs::Output *grey_frame_output, *tracked_frame_output, *pose_output, *pointcloud_output;
cv::Mat* cv_img;
float default_settings_scaledVarTH = 0.001, default_settings_absVarTH = 0.001, default_settings_minRelBS = 0.1;
int default_settings_pointCloudMode = 1, default_settings_sparsity = 1;

void set_mode()
{

    if(mode==0)
        printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");

    if(mode==1)
    {
        printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
        dso::setting_photometricCalibration = 0;
        dso::setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        dso::setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if(mode==2)
    {
        printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
        dso::setting_photometricCalibration = 0;
        dso::setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
        dso::setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
        dso::setting_minGradHistAdd=3;
    }
}

void set_outputs(SLAMBenchLibraryHelper * slam_settings)
{
    pose_output = new slambench::outputs::Output("DSO Pose", slambench::values::VT_POSE, true);

    pointcloud_output = new slambench::outputs::Output("DSO PointCloud", slambench::values::VT_POINTCLOUD, true);
    pointcloud_output->SetKeepOnlyMostRecent(true);

    grey_frame_output = new slambench::outputs::Output("DSO Grey Frame", slambench::values::VT_FRAME);
    grey_frame_output->SetKeepOnlyMostRecent(true);

    tracked_frame_output = new slambench::outputs::Output("DSO Tracked Frame", slambench::values::VT_FRAME);
    tracked_frame_output->SetKeepOnlyMostRecent(true);

    slam_settings->GetOutputManager().RegisterOutput(pose_output);
    slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);
    slam_settings->GetOutputManager().RegisterOutput(grey_frame_output);
    slam_settings->GetOutputManager().RegisterOutput(tracked_frame_output);
}

bool set_sensors(SLAMBenchLibraryHelper * slam_settings)
{
    slambench::io::CameraSensorFinder sensor_finder;
    grey_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "grey"}});
    if(grey_sensor == nullptr)
    {
        std::cerr << "Invalid sensors found, Grey not found." << std::endl;
        return false;
    }
    if(grey_sensor->PixelFormat != slambench::io::pixelformat::G_I_8)
    {
        std::cerr << "Grey sensor is not in G_I_8 format" << std::endl;
        return false;
    }
    if(grey_sensor->FrameFormat != slambench::io::frameformat::Raster)
    {
        std::cerr << "Grey sensor is not in raster format" << std::endl;
        return false;
    }

    return true;
}

bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings)
{
    slam_settings->addParameter(TypedParameter<int>("m", "mode", "Mode (add description here)", &mode, &default_mode));
    slam_settings->addParameter(TypedParameter<bool>("mt", "multithreading", "Multithreading (enabled by default)", &dso::multiThreading, nullptr));
    slam_settings->addParameter(TypedParameter<std::string>("v", "vignette", "Vignette file path", &vignetteFile, &default_vignette));
    slam_settings->addParameter(TypedParameter<std::string>("gc", "gammaCalib", "Gamma calibration file path", &gammaCalibFile, &default_gammaCalib));
    slam_settings->addParameter(TypedParameter<std::string>("ca", "calib", "Camera calibration", &calibFile, &default_calibFile));

    slam_settings->addParameter(TypedParameter<float>("scaled_TH", "", "Camera calibration", &settings_scaledVarTH, &default_settings_scaledVarTH));
    slam_settings->addParameter(TypedParameter<float>("abs_TH", "", "Camera calibration", &settings_absVarTH, &default_settings_absVarTH));
    slam_settings->addParameter(TypedParameter<int>("pcl_mode", "", "Camera calibration", &settings_pointCloudMode, &default_settings_pointCloudMode));
    slam_settings->addParameter(TypedParameter<float>("minRelBS", "", "Camera calibration", &settings_minRelBS, &default_settings_minRelBS));
    slam_settings->addParameter(TypedParameter<int>("sparsity", "", "Camera calibration", &settings_sparsity, &default_settings_sparsity));
    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)
{
    assert(set_sensors(slam_settings));
    set_outputs(slam_settings);
    set_mode();
    cv_img = new cv::Mat(grey_sensor->Height, grey_sensor->Width,CV_8UC1);

    undistorter = Undistort::getUndistorterForFile(calibFile, gammaCalibFile, vignetteFile);
    Eigen::Matrix3f K = undistorter->getK().cast<float>();
    dso::setGlobalCalib(undistorter->getSize()[0], undistorter->getSize()[1], K);

    fullSystem = new FullSystem();
    fullSystem->linearizeOperation = true;
    fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
    wrapper = new dso::IOWrap::SLAMBenchOutputWrapper(grey_sensor->Width, grey_sensor->Height);
    fullSystem->outputWrapper.push_back(wrapper);

    return !fullSystem->initFailed;
}

static bool frame_ready = false;
bool sb_update_frame (SLAMBenchLibraryHelper *, slambench::io::SLAMFrame* s)
{
    if(s->FrameSensor == grey_sensor)
    {
        memcpy(cv_img->data, s->GetData(), s->GetSize());
        MinimalImageB minImg((int)cv_img->cols, (int)cv_img->rows, (unsigned char*)cv_img->data);
//      FIXME: need proper exposure here
        img = undistorter->undistort<unsigned char>(&minImg, 1, s->Timestamp.ToS());
        img->timestamp = s->Timestamp.ToS();
        s->FreeData();
        frame_ready = true;
        return true;
    }

    return false;
}

bool sb_process_once (SLAMBenchLibraryHelper *)
{
    static int frame = 0;
    if(frame_ready)
    {
        fullSystem->addActiveFrame(img, frame);
        frame++;
        delete img;
        frame_ready = false;
    }
    return true;
}

bool sb_get_tracked(bool* tracked)
{
    *tracked = !fullSystem->isLost;
    return *tracked;
}

bool sb_clean_slam_system()
{
    fullSystem->blockUntilMappingIsFinished();
    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }
    delete fullSystem;
    delete undistorter;
    return true;
}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *ts)
{
    (void)lib;
    (void)ts;

    if(pose_output->IsActive())
    {
        Eigen::Matrix4f matrix = wrapper->getCurrentPose().matrix().cast<float>();
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pose_output->AddPoint(*ts, new slambench::values::PoseValue(matrix));
    }

    if(pointcloud_output->IsActive())
    {
        std::vector<Vec3f> vector_cloud = wrapper->getCurrentCloud();
        slambench::values::PointCloudValue *point_cloud = new slambench::values::PointCloudValue();

        for(auto &point : vector_cloud)
        {
            point_cloud->AddPoint(slambench::values::Point3DF(point[0], point[1], point[2]));
        }
//         Take lock only after generating the map
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pointcloud_output->AddPoint(*ts, point_cloud);
    }

    if(grey_frame_output->IsActive())
    {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        grey_frame_output->AddPoint(*ts, new slambench::values::FrameValue(grey_sensor->Width, grey_sensor->Height, slambench::io::pixelformat::G_I_8, (void*)(cv_img->data)));
    }

    if(tracked_frame_output->IsActive())
    {

        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        tracked_frame_output->AddPoint(*ts, new slambench::values::FrameValue(grey_sensor->Width, grey_sensor->Height, slambench::io::pixelformat::G_I_8, (void*)(cv_img->data)));
    }

    return true;
}

bool sb_relocalize(SLAMBenchLibraryHelper *lib)
{
    return true;
}