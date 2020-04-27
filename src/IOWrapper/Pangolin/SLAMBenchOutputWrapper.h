/*
 * PangolinOutput3DWrapper.h
 *
 *  Created on: 17 Oct 2014
 *      Author: thomas
 */

#ifndef PANGOLINOUTPUT3DWRAPPER_H_
#define PANGOLINOUTPUT3DWRAPPER_H_
#include <pangolin/pangolin.h>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <map>
#include <deque>


namespace dso
{

    class FrameHessian;
    class CalibHessian;
    class FrameShell;


    namespace IOWrap
    {

        class KeyFrameDisplay;

        struct GraphConnection
        {
            KeyFrameDisplay* from;
            KeyFrameDisplay* to;
            int fwdMarg, bwdMarg, fwdAct, bwdAct;
        };


        class SLAMBenchOutputWrapper : public Output3DWrapper
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            SLAMBenchOutputWrapper(int w, int h);
            virtual ~SLAMBenchOutputWrapper();

            void run();
            void close();

            void addImageToDisplay(std::string name, MinimalImageB3* image);
            void clearAllImagesToDisplay();


            // ==================== Output3DWrapper Functionality ======================
            virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
            virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override;
            virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;


            virtual void pushLiveFrame(FrameHessian* image) override;
            virtual void pushDepthImage(MinimalImageB3* image) override;
            virtual bool needPushDepthImage() override;

            SE3 getCurrentPose() const;
            std::vector<Vec3f> getCurrentCloud() const;
        private:

            bool needReset;
            void reset_internal();

            int w,h;



            // images rendering
            boost::mutex openImagesMutex;
            MinimalImageB3* internalVideoImg;
            MinimalImageB3* internalKFImg;
            MinimalImageB3* internalResImg;
            bool videoImgChanged, kfImgChanged, resImgChanged;



            // 3D model rendering
            boost::mutex model3DMutex;
            KeyFrameDisplay* currentCam;
            std::vector<KeyFrameDisplay*> keyframes;
            std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;
            std::map<int, KeyFrameDisplay*> keyframesByKFID;
            std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>> connections;
            SE3 currentPose, lastKeyframePose;


            // render settings
            bool settings_showKFCameras;
            bool settings_showCurrentCamera;
            bool settings_showTrajectory;
            bool settings_showFullTrajectory;
            bool settings_showActiveConstraints;
            bool settings_showAllConstraints;

            // timings
            struct timeval last_track;
            struct timeval last_map;


            std::deque<float> lastNTrackingMs;
            std::deque<float> lastNMappingMs;
        };



    }



}

#endif /* PANGOLINOUTPUT3DWRAPPER_H_ */
