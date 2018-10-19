//
// Created by sebastiano on 8/18/16.
//

#ifndef ORB_SLAM2_ROSPUBLISHER_H
#define ORB_SLAM2_ROSPUBLISHER_H

#include "IPublisherThread.h"
#include "IFrameSubscriber.h"
#include "IMapPublisher.h"
#include "FrameDrawer.h"
#include "System.h"
#include "Map.h"

#include <chrono>
#include <mutex>

#include <ros/ros.h>
#include <ros/time.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>

#include <orb_slam2_ros/ORBState.h>

namespace ORB_SLAM2
{
    class Map;
    class Tracking;
    class LoopClosing;
}

class ROSPublisher :
    public ORB_SLAM2::IPublisherThread,
    public ORB_SLAM2::IMapPublisher,
    public ORB_SLAM2::IFrameSubscriber
{
public:
    static constexpr const char *DEFAULT_MAP_FRAME = "/orb_slam2/world";
    static constexpr const char *DEFAULT_MAP_FRAME_ADJUSTED = "/orb_slam2/map";
    static constexpr const char *DEFAULT_CAMERA_FRAME = "/orb_slam2/camera";
    static constexpr const char *DEFAULT_BASE_FRAME = "ORB_base_link";
    static constexpr const float DEFAULT_OCTOMAP_RESOLUTION = 0.1;
    static constexpr const float ORBSTATE_REPUBLISH_RATE = 20;    // re-publish state @ 20 Hz
    static constexpr const float PROJECTION_MIN_HEIGHT = -10;
    static constexpr const float GRADIENT_MAX_HEIGHT = 0;        // maximal voxel-z to consider in gradient-based projection
    static constexpr const int   GRADIENT_NB_EROSIONS = 1;        // number of erosions performed before computing gradients
    static constexpr const float GRADIENT_LOW_SLOPE = M_PI / 4.;  // lower bound for a slope being considered obstacle-ish
    static constexpr const float GRADIENT_HIGH_SLOPE = M_PI / 3.; // lower bound for a slope being considered a full solid obstacle
    static constexpr const float OCTOMAP_RATE = 1;                // rate of octomap cycles (integrate MapPoints and publish)
    static constexpr const bool  OCTOMAP_REBUILD = false;         // clear octomap when tracking is lost and rebuild

    // `frequency` is max amount of messages emitted per second
    explicit ROSPublisher(
        ORB_SLAM2::Map *map,
        double frequency,
        ros::NodeHandle nh = ros::NodeHandle());

    virtual void Run() override;
    virtual void Update(ORB_SLAM2::Tracking*);

protected:
    bool WaitCycleStart();

private:
    ORB_SLAM2::FrameDrawer drawer_;

    // Important: `nh_` goes before the `*_pub_`, because their construction relies on `nh_`!
    ros::NodeHandle nh_;
    ros::Publisher map_pub_, map_updates_pub_, image_pub_, odom_pub_, pose_pub_, state_pub_, state_desc_pub_,
    projected_map_pub_, gradient_map_pub_, waypoint_pub_;
    tf::TransformBroadcaster camera_tf_pub_;
    ros::Rate pub_rate_;
    ros::ServiceClient localization_client_;

    ros::Time last_state_publish_time_;
    orb_slam2_ros::ORBState orb_state_;

    int lastBigMapChange_;
    tf::Vector3 camera_position_;

    int prev_closes_;
    bool localization_mode_;

    int publish_rate_;
    bool twoD;
    bool publish_tf_;
    float waypoint_z_thresh_;

    // frame data
    std::string map_frame_;
    std::string map_frame_adjusted_;
    std::string odom_frame_;
    std::string camera_frame_;
    std::string base_frame_;

    std::mutex pointcloud_map_points_mutex_;
    int pointcloud_chunks_stashed_;

    // params for z-plane-based occupancy grid approach
    double projection_min_height_;
    // params for gradient-based approach
    float gradient_max_height_;
    int gradient_nb_erosions_;
    float gradient_low_slope_;
    float gradient_high_slope_;

    tf::TransformListener tf_listener_;

    void publishMap();
    void publishMapUpdates();
    void publishCameraPose();
    void publishState(ORB_SLAM2::Tracking *tracking);
    void publishImage(ORB_SLAM2::Tracking *tracking);
    void publishProjectedMap();
    void publishGradientMap();
    visualization_msgs::Marker getWaypoints(ORB_SLAM2::Tracking *tracking);
};

class ROSSystemBuilder : public ORB_SLAM2::System::GenericBuilder {
public:
    ROSSystemBuilder(const std::string& strVocFile,
                     const std::string& strSettingsFile,
                     ORB_SLAM2::System::eSensor sensor,
                     double frequency,
                     ros::NodeHandle nh = ros::NodeHandle(),
                     std::string map_frame = ROSPublisher::DEFAULT_MAP_FRAME,
                     std::string camera_frame = ROSPublisher::DEFAULT_CAMERA_FRAME);
    virtual ~ROSSystemBuilder();
    
    virtual ORB_SLAM2::IPublisherThread* GetPublisher() override;

private:
    std::unique_ptr<ROSPublisher> mpPublisher;
};


#endif //ORB_SLAM2_ROSPUBLISHER_H
