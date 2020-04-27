#include "SLAMBenchOutputWrapper.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "KeyFrameDisplay.h"
#include "util/globalCalib.h"
#include "util/settings.h"

using namespace dso;
using namespace dso::IOWrap;

SLAMBenchOutputWrapper::SLAMBenchOutputWrapper(int w, int h)
{
    this->w = w;
    this->h = h;

    {
        boost::unique_lock<boost::mutex> lk(openImagesMutex);
        internalVideoImg = new MinimalImageB3(w,h);
        internalKFImg = new MinimalImageB3(w,h);
        internalResImg = new MinimalImageB3(w,h);
        videoImgChanged=kfImgChanged=resImgChanged=true;

        internalVideoImg->setBlack();
        internalKFImg->setBlack();
        internalResImg->setBlack();
    }

    currentCam = new KeyFrameDisplay();
    needReset = false;
}

SLAMBenchOutputWrapper::~SLAMBenchOutputWrapper() {}

void SLAMBenchOutputWrapper::reset_internal()
{
    model3DMutex.lock();
    for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
    keyframes.clear();
    allFramePoses.clear();
    keyframesByKFID.clear();
    connections.clear();
    model3DMutex.unlock();


    openImagesMutex.lock();
    internalVideoImg->setBlack();
    internalKFImg->setBlack();
    internalResImg->setBlack();
    videoImgChanged = kfImgChanged = resImgChanged = true;
    openImagesMutex.unlock();

    needReset = false;
}

void SLAMBenchOutputWrapper::publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

    model3DMutex.lock();
    connections.resize(connectivity.size());
    int runningID=0;
    int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
    {
        int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

        assert(host >= 0 && target >= 0);
        if(host == target)
        {
            assert(p.second[0] == 0 && p.second[1] == 0);
            continue;
        }

        if(host > target) continue;

        connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
        connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
        connections[runningID].fwdAct = p.second[0];
        connections[runningID].fwdMarg = p.second[1];
        totalActFwd += p.second[0];
        totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
        Eigen::Vector2i st = connectivity.at(inverseKey);
        connections[runningID].bwdAct = st[0];
        connections[runningID].bwdMarg = st[1];

        totalActBwd += st[0];
        totalMargBwd += st[1];

        runningID++;
    }


    model3DMutex.unlock();
}
void SLAMBenchOutputWrapper::publishKeyframes(
    std::vector<FrameHessian*> &frames,
    bool final,
    CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

    boost::unique_lock<boost::mutex> lk(model3DMutex);
    for(FrameHessian* fh : frames)
    {
        if(keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
        {
            KeyFrameDisplay* kfd = new KeyFrameDisplay();
            keyframesByKFID[fh->frameID] = kfd;
            keyframes.push_back(kfd);
        }
        keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
    }
}
void SLAMBenchOutputWrapper::publishCamPose(FrameShell* frame,
                                             CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

    boost::unique_lock<boost::mutex> lk(model3DMutex);
    struct timeval time_now;
    gettimeofday(&time_now, NULL);
    lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
    if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
    last_track = time_now;

    if(!setting_render_display3D) return;

    currentCam->setFromF(frame, HCalib);
    allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
    currentPose = frame->camToWorld;
}


void SLAMBenchOutputWrapper::pushLiveFrame(FrameHessian* image)
{
    if(!setting_render_displayVideo) return;
    if(disableAllDisplay) return;

//    boost::unique_lock<boost::mutex> lk(openImagesMutex);
//
//    for(int i=0;i<w*h;i++)
//        internalVideoImg->data[i][0] =
//        internalVideoImg->data[i][1] =
//        internalVideoImg->data[i][2] =
//                image->dI[i][0]*0.8 > 255.0f ? 255.0 : image->dI[i][0]*0.8;
//
//    videoImgChanged=true;
}

bool SLAMBenchOutputWrapper::needPushDepthImage()
{
    return setting_render_displayDepth;
}

void SLAMBenchOutputWrapper::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth) return;
    if(disableAllDisplay) return;

    boost::unique_lock<boost::mutex> lk(openImagesMutex);

    struct timeval time_now;
    gettimeofday(&time_now, NULL);
    lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
    if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
    last_map = time_now;

    memcpy(internalKFImg->data, image->data, w*h*3);
    kfImgChanged=true;
}

SE3 SLAMBenchOutputWrapper::getCurrentPose() const
{
    return currentPose;
}


std::vector<Vec3f> SLAMBenchOutputWrapper::getCurrentCloud() const
{
    std::vector<Vec3f> point_cloud;
    int refreshed=0;
    int pointindex = 0;
    for(KeyFrameDisplay* kf : keyframes)
    {
        Eigen::Matrix4f mat = kf->camToWorld.matrix().cast<float>();
        refreshed += (int)(kf->refreshPC(refreshed < 10,
                                         settings_scaledVarTH,
                                         settings_absVarTH,
                                         settings_pointCloudMode,
                                         settings_minRelBS,
                                         settings_sparsity));
        auto kf_cloud = kf->getPC();
        auto size = kf->getPCSize();
        point_cloud.resize(point_cloud.size()+size);

        for(int i = 0; i < size; i++)
        {
            Eigen::Vector4f original = {kf_cloud[i][0], kf_cloud[i][1], kf_cloud[i][2], 1};
            original = mat * original;
            point_cloud[pointindex][0] = original[0] / original[3];
            point_cloud[pointindex][1] = original[1] / original[3];
            point_cloud[pointindex][2] = original[2] / original[3];
            pointindex++;
        }
    }


    return point_cloud;
}
