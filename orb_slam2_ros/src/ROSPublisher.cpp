//
// Created by sebastiano on 8/18/16.
//

#include "ROSPublisher.h"
#include "FrameDrawer.h"
#include "Tracking.h"
#include "LoopClosing.h"
#include "utils.h"

#include <thread>
#include <sstream>
#include <cassert>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/LinearMath/Transform.h>
#include "tf/transform_broadcaster.h"
#include "tf/transform_listener.h"
#include "tf/message_filter.h"
#include "tf/tf.h"
#include "message_filters/subscriber.h"
#include <visualization_msgs/Marker.h>
#include <std_srvs/SetBool.h>


#include "tf/transform_datatypes.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include <std_msgs/String.h>
#include <orb_slam2_ros/ORBState.h>
#include <cv_bridge/cv_bridge.h>



#include <chrono>


using namespace ORB_SLAM2;


static bool isBigEndian() {
    volatile int num = 1;
    return *((char *) &num) == ((char) 1);
}

static const bool IS_BIG_ENDIAN = isBigEndian();

namespace std {
    std::string to_string(const cv::Mat &mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
    }
}

template<typename Q>
Q convertToQuaternion(const cv::Mat &rot) {
    double trace = rot.at<float>(0, 0) + rot.at<float>(1, 1) + rot.at<float>(2, 2);
    double tmp[4];

    if (trace > 0.0) {
        double s = sqrt(trace + 1.0);
        tmp[3] = s * 0.5;
        s = 0.5 / s;
        tmp[0] = ((rot.at<float>(2, 1) - rot.at<float>(1, 2)) * s);
        tmp[1] = ((rot.at<float>(0, 2) - rot.at<float>(2, 0)) * s);
        tmp[2] = ((rot.at<float>(1, 0) - rot.at<float>(0, 1)) * s);
    } else {
        int i;
        if (rot.at<float>(0, 0) < rot.at<float>(1, 1))
            i = rot.at<float>(1, 1) < rot.at<float>(2, 2) ? 2 : 1;
        else
            i = rot.at<float>(0, 0) < rot.at<float>(2, 2) ? 2 : 0;
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        double s = sqrt(rot.at<float>(i, i) - rot.at<float>(j, j) - rot.at<float>(k, k) + 1.0);
        tmp[i] = s * 0.5;
        s = 0.5 / s;
        tmp[3] = (rot.at<float>(k, j) - rot.at<float>(j, k)) * s;
        tmp[j] = (rot.at<float>(j, i) + rot.at<float>(i, j)) * s;
        tmp[k] = (rot.at<float>(k, i) + rot.at<float>(i, k)) * s;
    }

    return {tmp[0], tmp[1], tmp[2], tmp[3]};
}

cv::Mat computeCameraTransform(const cv::Mat &Twc) {
    cv::Mat ret = cv::Mat::eye(4, 4, CV_32F);

    if (!Twc.empty()) {
        auto Rwc = Twc.rowRange(0, 3).colRange(0, 3).t();
        ret.rowRange(0, 3).colRange(0, 3) = Rwc;
        // twc, the position
        ret.rowRange(0, 3).col(3) = -Rwc * Twc.rowRange(0, 3).col(3);
    }
    return ret;
}

sensor_msgs::PointCloud2 convertToPCL2(const std::vector<MapPoint *> &map_points) {
    const std::size_t n_map_points = map_points.size();
    ROS_INFO("sending PointCloud (%lu points)", n_map_points);

    // Kind of a hack, but there aren't much better ways to avoid a copy
    struct point {
        float x, y, z;
    };

    std::vector<uint8_t> data_buffer(n_map_points * sizeof(point));
    std::size_t vtop = 0;

    point *dataptr = (point *) data_buffer.data();

    // also convert to normal ROS transforms
    for (MapPoint *map_point : map_points) {
        if (map_point->isBad())
            continue;
        cv::Mat pos = map_point->GetWorldPos();
        dataptr[vtop++] = {
                pos.at<float>(2),
                -pos.at<float>(0),
                -pos.at<float>(1),
        };
    }

    static const char *const names[3] = {"x", "y", "z"};
    static const std::size_t offsets[3] = {offsetof(point, x), offsetof(point, y), offsetof(point, z)};
    std::vector<sensor_msgs::PointField> fields(3);
    for (int i = 0; i < 3; i++) {
        fields[i].name = names[i];
        fields[i].offset = offsets[i];
        fields[i].datatype = sensor_msgs::PointField::FLOAT32;
        fields[i].count = 1;
    }

    sensor_msgs::PointCloud2 msg;
    msg.height = 1;
    msg.width = n_map_points;
    msg.fields = fields;
    msg.is_bigendian = IS_BIG_ENDIAN;
    msg.point_step = sizeof(point);
    msg.row_step = sizeof(point) * msg.width;
    msg.data = std::move(data_buffer);
    msg.is_dense = true;  // invalid points already filtered out

    return msg;
}

/*
 * Returns a ROS parameter as generic type, defaulting to a given value if it is unspecified.
 */
template<typename T>
T getROSParam(ros::NodeHandle nh, std::string param_name, T default_value) {
    T result;
    nh.param<T>(param_name, result, default_value);
    return result;
}


ROSPublisher::ROSPublisher(Map *map, double frequency, ros::NodeHandle nh) :
        IMapPublisher(map),
        drawer_(GetMap()),
        nh_("~"),
        pub_rate_(frequency),
        lastBigMapChange_(-1),
        pointcloud_chunks_stashed_(0),
        localization_mode_(false),
        prev_closes_(0) {
    orb_state_.state = orb_slam2_ros::ORBState::UNKNOWN;

    // initialize parameters
    nh_.param<int>("publish_rate", publish_rate_, 20);
    nh_.param<bool>("twoD", twoD, false);
    nh_.param<bool>("publish_tf", publish_tf_, false);

    nh_.param<double>("occupancy_projection_min_height", projection_min_height_, ROSPublisher::PROJECTION_MIN_HEIGHT);

    nh_.param<float>("occupancy_gradient_max_height", gradient_max_height_, ROSPublisher::GRADIENT_MAX_HEIGHT);
    nh_.param<int>("occupancy_gradient_nb_erosions", gradient_nb_erosions_, ROSPublisher::GRADIENT_NB_EROSIONS);
    nh_.param<float>("occupancy_gradient_low_slope", gradient_low_slope_, ROSPublisher::GRADIENT_LOW_SLOPE);
    nh_.param<float>("occupancy_gradient_high_slope", gradient_high_slope_, ROSPublisher::GRADIENT_HIGH_SLOPE);

    nh_.param<float>("waypoint_z_threshold", waypoint_z_thresh_, 1.5);

    nh_.param<std::string>("map_frame", map_frame_, "map");
    nh_.param<std::string>("map_frame_adjusted", map_frame_adjusted_, "map");
    nh_.param<std::string>("odom_frame", odom_frame_, "odom");
    nh_.param<std::string>("camera_frame", camera_frame_, "base_footprint");
    nh_.param<std::string>("base_frame", base_frame_, "base_footprint");

    // initialize publishers
    // TODO make more params configurable
    map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map", 3);
    map_updates_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_updates", 3);
    image_pub_ = nh_.advertise<sensor_msgs::Image>("frame", 5);
    state_pub_ = nh_.advertise<orb_slam2_ros::ORBState>("state", 10);
    state_desc_pub_ = nh_.advertise<std_msgs::String>("state_description", 10);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 10);
    waypoint_pub_ = nh_.advertise<visualization_msgs::Marker>("waypoints", 10);

    localization_client_ = nh_.serviceClient<std_srvs::SetBool>("localization");

}



static const char *stateDescription(orb_slam2_ros::ORBState orb_state) {
    switch (orb_state.state) {
        case orb_slam2_ros::ORBState::SYSTEM_NOT_READY:
            return "System not ready";
        case orb_slam2_ros::ORBState::NO_IMAGES_YET:
            return "No images yet";
        case orb_slam2_ros::ORBState::NOT_INITIALIZED:
            return "Not initialized";
        case orb_slam2_ros::ORBState::LOCALIZING:
                return "LOCALIZING";
        case orb_slam2_ros::ORBState::MAPPING:
                return "MAPPING";
        case orb_slam2_ros::ORBState::LOST:
            return "Tracking lost";
    }

    return "???";
}

static const orb_slam2_ros::ORBState toORBStateMessage(Tracking::eTrackingState trackingState, bool localization_mode) {
    orb_slam2_ros::ORBState state_msg;
    state_msg.state = orb_slam2_ros::ORBState::UNKNOWN;

    switch (trackingState) {
        case Tracking::SYSTEM_NOT_READY:
            state_msg.state = orb_slam2_ros::ORBState::SYSTEM_NOT_READY;
            break;
        case Tracking::NO_IMAGES_YET:
            state_msg.state = orb_slam2_ros::ORBState::NO_IMAGES_YET;
            break;
        case Tracking::NOT_INITIALIZED:
            state_msg.state = orb_slam2_ros::ORBState::NOT_INITIALIZED;
            break;
        case Tracking::OK:
            if (localization_mode)
                state_msg.state = orb_slam2_ros::ORBState::LOCALIZING;
            else
                state_msg.state = orb_slam2_ros::ORBState::MAPPING;
            break;
        case Tracking::LOST:
            state_msg.state = orb_slam2_ros::ORBState::LOST;
            break;
    }

    return state_msg;
}

/*
 * Publishes ORB_SLAM 2 GetAllMapPoints() as a PointCloud2.
 */
void ROSPublisher::publishMap() {
    if (map_pub_.getNumSubscribers() > 0) {
        auto msg = convertToPCL2(GetMap()->GetAllMapPoints());
        msg.header.frame_id = map_frame_;
        map_pub_.publish(msg);
    }
}

/*
 * Publishes ORB_SLAM 2 GetReferenceMapPoints() as a PointCloud2.
 */
void ROSPublisher::publishMapUpdates() {
    if (map_updates_pub_.getNumSubscribers() > 0) {
        auto msg = convertToPCL2(GetMap()->GetReferenceMapPoints());
        msg.header.frame_id = map_frame_;
        map_updates_pub_.publish(msg);
    }
}

/*
 * Publishes ORB_SLAM 2 GetCameraPose() as a TF.
 */
void ROSPublisher::publishCameraPose() {
    // number of subscribers is unknown to a TransformBroadcaster
    cv::Mat xf = computeCameraTransform(GetCameraPose());
    if (!xf.empty()) {

        // get standard orientation of camera
        auto orientation = convertToQuaternion<tf::Quaternion>(xf);

        if (twoD) {
            camera_position_ = {xf.at<float>(2, 3), -xf.at<float>(0, 3), 0};

            tf::Matrix3x3 m(orientation);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);

            // convert only yaw back to quaternion
            orientation = tf::createQuaternionFromYaw(yaw);

            // apply orbslam bias from the OpenCV coordinate frame
            auto orbslam_bias = tf::createQuaternionFromRPY(0, 0, 0);
            orientation = orientation * orbslam_bias;
        } else {
            camera_position_ = {xf.at<float>(2, 3), -xf.at<float>(0, 3), -xf.at<float>(1, 3)};

            // apply orbslam bias from the OpenCV coordinate frame
            auto orbslam_bias = tf::createQuaternionFromRPY(0, 0, 0);
            orientation = orientation * orbslam_bias;
        }


        if (publish_tf_) {

            tf::Stamped<tf::Pose> odom_to_map;

            tf::Transform tmp(orientation, camera_position_);

            try {
                // create camera to map tf

                tf::StampedTransform base_to_odom_tf;
                tf_listener_.lookupTransform(odom_frame_, base_frame_, ros::Time(0), base_to_odom_tf);
                tf::Stamped<tf::Pose> camera_pose(tmp.inverse(),
                                                  base_to_odom_tf.stamp_, base_frame_);

                tf_listener_.transformPose(odom_frame_, camera_pose, odom_to_map);


                auto tmp_tf = tf::Transform(tf::Quaternion(odom_to_map.getRotation()),
                                            tf::Point(odom_to_map.getOrigin()));

                tf::StampedTransform map_to_odom_tf(tmp_tf.inverse(), base_to_odom_tf.stamp_, map_frame_, odom_frame_);

                camera_tf_pub_.sendTransform(map_to_odom_tf);

            }
            catch (tf::TransformException) {
                ROS_WARN("Failed to subtract base to odom transform");
            }
        }

        // create pose message
        auto pose = geometry_msgs::PoseWithCovarianceStamped();
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = map_frame_;

        pose.pose.pose.position.x = camera_position_.x();
        pose.pose.pose.position.y = camera_position_.y();
        pose.pose.pose.position.z = camera_position_.z();

        pose.pose.pose.orientation.z = orientation.z();
        pose.pose.pose.orientation.y = orientation.y();
        pose.pose.pose.orientation.x = orientation.x();
        pose.pose.pose.orientation.w = orientation.w();

        pose.pose.covariance[0] = 10e-9;
        pose.pose.covariance[7] = 10e-9;
        pose.pose.covariance[35] = 0.1;

        pose_pub_.publish(pose);

        ResetCamFlag();
    }
}

/*
 * Publishes the ORB_SLAM 2 tracking state as ORBState int and/or as a description string.
 */
void ROSPublisher::publishState(Tracking *tracking) {

    last_state_publish_time_ = ros::Time::now();

    if (tracking == NULL) {
        return;
    }

    // save state from tracking, even if there are no subscribers
    orb_state_ = toORBStateMessage(tracking->mState, localization_mode_);

    if (state_pub_.getNumSubscribers() > 0) {
        // publish state as ORBState int
        orb_state_.header.stamp = ros::Time::now();
        orb_state_.num_closes = tracking->getLoopCloses();
        state_pub_.publish(orb_state_);
    }

    if (state_desc_pub_.getNumSubscribers() > 0) {
        // publish state as string
        std_msgs::String state_desc_msg;
        state_desc_msg.data = stateDescription(orb_state_) + std::to_string(tracking->getLoopCloses());
        state_desc_pub_.publish(state_desc_msg);
    }


    // if the map has loop closed, then publish new waypoints
    if (prev_closes_ < tracking->getLoopCloses()) {
        prev_closes_ = tracking->getLoopCloses();
        waypoint_pub_.publish(getWaypoints(tracking));

//        std_srvs::SetBool srv;
//        srv.request.data = true;
//        if (localization_client_.call(srv)) {
//            tracking->ActivateLocalizationMode();
//        }

        // TODO remove this once the monitor is finished
        tracking->ActivateLocalizationMode();
        localization_mode_ = true;
    }
}

/*
 * Publishes the current ORB_SLAM 2 status image.
 */
void ROSPublisher::publishImage(Tracking *tracking) {
//    std::cout << "publishing image" << std::endl;
    if (image_pub_.getNumSubscribers() > 0) {
        drawer_.Update(tracking);

        std_msgs::Header hdr;
        cv_bridge::CvImage cv_img{hdr, "bgr8", drawer_.DrawFrame()};

        auto image_msg = cv_img.toImageMsg();
        image_msg->header = hdr;
        image_pub_.publish(*image_msg);
    }
}

visualization_msgs::Marker ROSPublisher::getWaypoints(Tracking *tracking) {
    // get the camera keyframes from the ORBSLAM algorithm

    // wait for tracking
//    while(tracking == NULL) {}

    cv::Mat camera_frames = tracking->getKeyFrames();

    visualization_msgs::Marker waypoints;

    std::cout << "Received " << camera_frames.rows << " camera frames" << std::endl;

    for (int i = 0; i < camera_frames.rows; i=i+8) {
        const float *Mi = camera_frames.ptr<float>(i);
        geometry_msgs::Point point;
        point.x = std::isnan(Mi[3]) ? 0 : Mi[3];
        point.y = std::isnan(-Mi[1]) ? 0 : -Mi[1];

        if (twoD) {
            point.z = 0;
        } else {
            point.z = std::isnan(-Mi[2]) ? 0 : -Mi[2];
        }

        waypoints.points.push_back(point);
    }

    // populate the rest of the needed data for the Marker message
    waypoints.header.frame_id = map_frame_;
    waypoints.header.stamp = ros::Time::now();
    waypoints.ns = "/eufsa";
    waypoints.id = 0;
    waypoints.type = visualization_msgs::Marker::POINTS;
    waypoints.action = visualization_msgs::Marker::ADD;
    waypoints.pose = geometry_msgs::Pose();
    waypoints.scale.x = 0.5;
    waypoints.scale.y = 0.5;
    waypoints.scale.z = 0.5;
    waypoints.color.a = 1.0;
    waypoints.color.r = 1.0;
    waypoints.color.g = 0.0;
    waypoints.color.b = 0.0;

    return waypoints;

}

void ROSPublisher::Run() {
    using namespace std::this_thread;
    using namespace std::chrono;

    SetFinish(false);

    ROS_INFO("ROS publisher started");

    while (WaitCycleStart()) {
        // only publish map, map updates and camera pose, if camera pose was updated
        // TODO: maybe there is a way to check if the map was updated
        if (isCamUpdated()) {
            publishMap();
            publishMapUpdates();
            publishCameraPose();

        }

        if (ros::Time::now() >= last_state_publish_time_ + ros::Duration(1. / publish_rate_)) {
            // it's time to re-publish ORBState
            publishState(NULL);
        }
    }

    ROS_INFO("ROS publisher finished");
    SetFinish(true);
}

bool ROSPublisher::WaitCycleStart() {
    //std::cout << "in WaitCycleStart" << std::endl;
    if (!IPublisherThread::WaitCycleStart())
        return false;

    pub_rate_.sleep();
    return true;
}

void ROSPublisher::Update(Tracking *tracking) {
//    std::cout << "Received new update" << std::endl;
    static std::mutex mutex;

    if (tracking == nullptr)
        return;


    publishState(tracking);


    // TODO: Make sure the camera TF is correctly aligned. See:
    // <http://docs.ros.org/jade/api/sensor_msgs/html/msg/Image.html>

    publishImage(tracking);

    // sends trajectory in real time; used for debugging
//    waypoint_pub_.publish(getWaypoints(tracking));
}

ROSSystemBuilder::ROSSystemBuilder(const std::string &strVocFile,
                                   const std::string &strSettingsFile,
                                   ORB_SLAM2::System::eSensor sensor,
                                   double frequency,
                                   ros::NodeHandle nh,
                                   std::string map_frame,
                                   std::string camera_frame) :
        System::GenericBuilder(strVocFile, strSettingsFile, sensor) {
    mpPublisher = make_unique<ROSPublisher>(
            GetMap(), frequency, nh);
    mpTracker->SetFrameSubscriber(mpPublisher.get());
    mpTracker->SetMapPublisher(mpPublisher.get());
}

// Empty dtor to give a place to the calls to the dtor of unique_ptr members
ROSSystemBuilder::~ROSSystemBuilder() {}

IPublisherThread *ROSSystemBuilder::GetPublisher() {
    return mpPublisher.get();
}