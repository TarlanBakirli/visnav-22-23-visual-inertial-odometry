/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>
#include <basalt/imu/imu_types.h>

using namespace visnav;
using namespace basalt;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
void load_imu_data(const std::string& path);
void load_gt_data_state(const std::string& path);
void load_gt_data_pose(const std::string& path);
bool next_step();
void optimize();
void compute_projections();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int current_frame;  // default: = 0
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

std::shared_ptr<std::thread> opt_thread;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

// loaded IMUData
// tbb::concurrent_unordered_map<Timestamp, IMUData> imu_measurements;
std::vector<basalt::ImuData<double>> imu_measurements;

// loaded GT state
tbb::concurrent_unordered_map<Timestamp, GT_State> gt_state_measurements;

// loaded GT pose
tbb::concurrent_unordered_map<Timestamp, GT_Pose> gt_pose_measurements;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// timestamps for all imu data
std::vector<Timestamp> imu_timestamps;

/// timestamps for all GT states
std::vector<Timestamp> gt_state_timestamps;

/// timestamps for all GT Poses
std::vector<Timestamp> gt_pose_timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// copy of landmarks for optimization in parallel thread
Landmarks landmarks_opt;

/// landmark positions that were removed from the current map
Landmarks old_landmarks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

/// For VIO project
/// Initialization
/// copy of IMUs for optimization in parallel thread
IMUs imus_opt;

Eigen::Matrix<double, 3, 1> vel_w_i_init;
Sophus::SE3d T_w_i_init;
IMU_MEAS imu_meas;
FRAME_STATE frame_states;
int new_frame = 0;
std::set<FrameId> buffer_frames;
int max_num_buffers = 3;
// FRAME_STATE keyframe_states;
basalt::IntegratedImuMeasurement<double> imu_meas_init(1403636580888555520,
                                                       Eigen::Vector3d::Zero(),
                                                       Eigen::Vector3d::Zero());

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel
// by switching the prefix from "ui" to "hidden" or vice verca. This way you
// can show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 7, 5,
                               20);  // default: 10, change for VIO

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/MH_01_easy/mav0";
  std::string cam_calib = "data/calibration.json";

  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib);
  load_imu_data(dataset_path);
  load_gt_data_state(dataset_path);
  load_gt_data_pose(dataset_path);

  // initialization
  initialize(imu_measurements, calib_cam, timestamps, frame_states);
  imu_meas[0] = imu_meas_init;

  return 0;
}

// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

// VIO Project: dataloader for IMUs and GT

void load_imu_data(const std::string& dataset_path) {
  imu_measurements.clear();

  const std::string timestams_path = dataset_path + "/imu0/data.csv";

  std::ifstream f(timestams_path);

  while (f) {
    std::string line;
    std::getline(f, line);

    if (line.size() < 20 || line[0] == '#') continue;

    {
      std::stringstream ss(line);

      char tmp;
      Timestamp timestamp;

      Eigen::Vector3d gyro, accel;

      ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
          tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];
      imu_timestamps.push_back(timestamp);

      basalt::ImuData<double> imudata;
      imudata.gyro = gyro;
      imudata.accel = accel;
      imudata.t_ns = timestamp;
      imu_measurements.push_back(imudata);
    }
  }

  std::cerr << "Loaded " << imu_measurements.size() << " IMUs" << std::endl;

  // calibration already read in load_data, no need here
}

void load_gt_data_state(const std::string& dataset_path) {
  const std::string timestams_path =
      dataset_path + "/state_groundtruth_estimate0/data.csv";
  std::ifstream f(timestams_path);

  while (f) {
    std::string line;
    std::getline(f, line);

    if (line.size() < 20 || line[0] == '#') continue;

    std::string timestamp_str = line.substr(0, 19);
    std::istringstream ss(timestamp_str);

    char tmp;
    Timestamp timestamp;
    Eigen::Quaterniond q;
    Eigen::Vector3d pos, vel, accel_bias, gyro_bias;

    ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >> tmp >>
        q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z() >> tmp >>
        vel[0] >> tmp >> vel[1] >> tmp >> vel[2] >> tmp >> accel_bias[0] >>
        tmp >> accel_bias[1] >> tmp >> accel_bias[2] >> tmp >> gyro_bias[0] >>
        tmp >> gyro_bias[1] >> tmp >> gyro_bias[2];

    gt_state_timestamps.push_back(timestamp);

    GT_State gt_state;
    gt_state.pos = pos;
    gt_state.q = q;
    gt_state.vel = vel;
    gt_state.accel_bias = accel_bias;
    gt_state.gyro_bias = gyro_bias;
    gt_state_measurements[timestamp] = gt_state;
  }
  std::cerr << "Loaded " << gt_state_measurements.size() << " GT states"
            << std::endl;
}

void load_gt_data_pose(const std::string& dataset_path) {
  const std::string timestams_path = dataset_path + "/leica0/data.csv";
  std::ifstream f(timestams_path);

  std::string line;
  while (std::getline(f, line)) {
    if (line.size() < 20 || line[0] == '#') continue;

    std::stringstream ss(line);

    char tmp;
    uint64_t timestamp;
    Eigen::Quaterniond q;
    Eigen::Vector3d pos;

    ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >> tmp >>
        q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z();

    gt_pose_timestamps.push_back(timestamp);

    GT_Pose gt_pose;
    gt_pose.pos = pos;
    gt_pose.q = q;
    gt_pose_measurements[timestamp] = gt_pose;
  }
  std::cerr << "Loaded " << gt_pose_measurements.size() << " GT poses"
            << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  std::cout << "FrameId of new frame: " << new_frame << std::endl;
  buffer_frames.emplace(new_frame);
  if (buffer_frames.size() > size_t(max_num_buffers))
    buffer_frames.erase(buffer_frames.begin());

  current_frame = *buffer_frames.rbegin();
  if (current_frame >= int(images.size()) / NUM_CAMS) return false;

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  // integrate for each current frame
  FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

  Timestamp curr_t_ns, last_t_ns;

  if (current_frame != 0) {
    curr_t_ns = timestamps[current_frame];
    last_t_ns = timestamps[current_frame - 1];

    integrate_imu(curr_t_ns, last_t_ns, imu_measurements,
                  imu_meas[current_frame], frame_states, current_frame);
  }

  if (take_keyframe) {
    take_keyframe = false;

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "KF Projected " << projected_track_ids.size() << " points."
              << std::endl;

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[fcidr]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << " stereo-matches."
              << std::endl;

    feature_corners[fcidl] = kdl;
    feature_corners[fcidr] = kdr;
    feature_matches[std::make_pair(fcidl, fcidr)] = md_stereo;

    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    cameras[fcidl].T_w_c = current_pose;
    cameras[fcidr].T_w_c = current_pose * T_0_1;

    add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md,
                      landmarks, next_landmark_id);

    // remove_old_keyframes(fcidl, max_num_kfs, cameras, landmarks,
    // old_landmarks, kf_frames);
    remove_old_keyframes_with_IMU(fcidl, max_num_kfs, cameras, landmarks,
                                  old_landmarks, kf_frames);
    optimize();

    current_pose = cameras[fcidl].T_w_c;

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    compute_projections();

    new_frame++;  // default: current_frame++
    return true;
  } else {
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "Projected " << projected_track_ids.size() << " points."
              << std::endl;

    KeypointsData kdl;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);

    feature_corners[fcidl] = kdl;

    LandmarkMatchData md;
    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    if (int(md.inliers.size()) < new_kf_min_inliers && !opt_running &&
        !opt_finished) {
      take_keyframe = true;
    }

    if (!opt_running && opt_finished) {
      opt_thread->join();
      landmarks = landmarks_opt;
      cameras = cameras_opt;
      calib_cam = calib_cam_opt;

      opt_finished = false;
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    new_frame++;  // default: current_frame++
    return true;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  size_t num_obs = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with " << cameras.size() << " cameras, "
            << landmarks.size() << " points and " << num_obs << " observations."
            << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole second
  // camera constant is a bit suboptimal, since we only need 1 DoF, but it's
  // simple and the initial poses should be good from calibration.
  FrameId fid = *(kf_frames.begin());
  // std::cout << "fid " << fid << std::endl;
  // std::cout << "kf_frames " << kf_frames.size() << std::endl;
  // for (auto it = kf_frames.begin(); it != kf_frames.end();
  //      it++)  // 使用迭代器进行遍历
  // {
  //   std::cout << "value of keyframe: " << *it << std::endl;
  // }

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  cameras_opt = cameras;
  landmarks_opt = landmarks;

  opt_running = true;

  opt_thread.reset(new std::thread([fid, ba_options] {
    std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};

    bundle_adjustment_with_IMU(feature_corners, ba_options, fixed_cameras,
                               calib_cam_opt, cameras_opt, landmarks_opt,
                               imu_meas, frame_states, kf_frames);  //, imus_opt

    opt_finished = true;
    opt_running = false;
  }));

  // Update project info cache
  compute_projections();
}
