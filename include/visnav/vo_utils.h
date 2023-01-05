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

#pragma once

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <basalt/imu/imu_types.h>
#include <basalt/imu/preintegration.h>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (const auto& landmark : landmarks) {
    Eigen::Vector3d p_3d = current_pose.inverse() * landmark.second.p;
    if (p_3d(2) >= cam_z_threshold) {
      Eigen::Vector2d p_2d = cam->project(p_3d);
      if (p_2d(0) >= 0 && p_2d(1) >= 0 && p_2d(0) <= cam->width() &&
          p_2d(1) <= cam->height()) {
        projected_points.push_back(p_2d);
        projected_track_ids.push_back(landmark.first);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.

  for (size_t featureId = 0; featureId < kdl.corners.size(); featureId++) {
    int dist;
    int smallest_dist = 256;
    int second_smallest_dist = 256;
    TrackId selected_track_id;
    for (size_t i = 0; i < projected_points.size(); i++) {
      if ((kdl.corners[featureId] - projected_points[i]).norm() <=
          match_max_dist_2d) {
        TrackId track_id = projected_track_ids[i];
        int landmark_dist = 256;
        for (const auto& obs : landmarks.at(track_id).obs) {
          dist =
              (kdl.corner_descriptors[featureId] ^
               feature_corners.at(obs.first).corner_descriptors.at(obs.second))
                  .count();
          if (dist < landmark_dist) {
            landmark_dist = dist;
          }
        }
        if (landmark_dist <= smallest_dist) {
          second_smallest_dist = smallest_dist;
          smallest_dist = landmark_dist;
          selected_track_id = track_id;
        } else if (landmark_dist < second_smallest_dist) {
          second_smallest_dist = landmark_dist;
        }
      }
    }
    if (second_smallest_dist >= smallest_dist * feature_match_dist_2_best &&
        smallest_dist < feature_match_threshold) {
      md.matches.push_back(std::make_pair(featureId, selected_track_id));
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly
  // have tracks.
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (const auto& match : md.matches) {
    const Eigen::Vector2d p_2d = kdl.corners[match.first];
    bearingVectors.push_back(cam->unproject(p_2d).normalized());
    points.push_back(landmarks.at(match.second).p);
  }

  // create the central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  // non-linear optimization (using all correspondences)
  adapter.setR(best_transformation.topLeftCorner(3, 3));
  adapter.sett(best_transformation.topRightCorner(3, 1).normalized());
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  for (const auto& inlier : ransac.inliers_)
    md.inliers.push_back(md.matches[inlier]);
  md.T_w_c = Sophus::SE3d(nonlinear_transformation.topLeftCorner(3, 3),
                          nonlinear_transformation.topRightCorner(3, 1));
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to
  // landmark matches add the observations to the existing landmarks. If the
  // left camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to
  // the existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a
  // new landmark you should always increase next_landmark_id by 1.

  for (const auto& md_inlier : md.inliers) {
    landmarks[md_inlier.second].obs[fcidl] = md_inlier.first;
    for (const auto& stereo_inlier : md_stereo.inliers) {
      if (md_inlier.first == stereo_inlier.first)
        landmarks[md_inlier.second].obs[fcidr] = stereo_inlier.second;
    }
  }

  // if exists in landmards
  for (const auto& stereo_inlier : md_stereo.inliers) {
    bool flag = 0;
    for (const auto& md_inlier : md.inliers) {
      if (md_inlier.first == stereo_inlier.first) {
        flag = 1;
      }
    }
    if (!flag) {
      opengv::bearingVectors_t bearingVectors1, bearingVectors2;
      bearingVectors1.push_back(calib_cam.intrinsics[fcidl.cam_id]->unproject(
          kdl.corners[stereo_inlier.first]));
      bearingVectors2.push_back(calib_cam.intrinsics[fcidr.cam_id]->unproject(
          kdr.corners[stereo_inlier.second]));
      opengv::relative_pose::CentralRelativeAdapter adapter(
          bearingVectors1, bearingVectors2, t_0_1, R_0_1);
      landmarks[next_landmark_id].p =
          md.T_w_c * opengv::triangulation::triangulate(adapter, 0);
      landmarks[next_landmark_id].obs.emplace(fcidl, stereo_inlier.first);
      landmarks[next_landmark_id].obs.emplace(fcidr, stereo_inlier.second);
      next_landmark_id++;
    }
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs. The ids of all the keyframes that are currently in the
  // optimization should be stored in kf_frames. Removed keyframes should be
  // removed from cameras and landmarks with no left observations should be
  // moved to old_landmarks.

  while (kf_frames.size() > size_t(max_num_kfs)) {
    FrameCamId old_fcidl = FrameCamId(*kf_frames.begin(), 0);
    FrameCamId old_fcidr = FrameCamId(*kf_frames.begin(), 1);
    cameras.erase(old_fcidl);
    cameras.erase(old_fcidr);
    std::vector<TrackId> track_ids;
    for (auto& landmark : landmarks) {
      landmark.second.obs.erase(old_fcidl);
      landmark.second.obs.erase(old_fcidr);
      if (landmark.second.obs.size() == 0) {
        TrackId track_id = landmark.first;
        old_landmarks[track_id] = landmarks[track_id];
        track_ids.push_back(track_id);
      }
    }
    for (const auto& track_id : track_ids) landmarks.erase(track_id);
    kf_frames.erase(kf_frames.begin());
  }
}

void integrate_imu(const Timestamp curr_t_ns, const Timestamp last_t_ns,
                   std::vector<basalt::ImuData<double>>& imu_measurements) {
  static const double accel_std_dev = 0.23;
  static const double gyro_std_dev = 0.0027;

  Eigen::Vector3d accel_cov, gyro_cov;
  accel_cov.setConstant(accel_std_dev * accel_std_dev);
  gyro_cov.setConstant(gyro_std_dev * gyro_std_dev);

  // replace these

  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  for (const auto& imudata : imu_measurements) {
    if (imudata.t_ns >= last_t_ns && imudata.t_ns <= curr_t_ns) {
      imu_meas.integrate(imudata, accel_cov, gyro_cov);
    }
  }
}

// Transf
void save_integrated_state(
    std::vector<basalt::ImuData<double>>& imu_measurements) {
  basalt::PoseVelState<double> state0;
  basalt::PoseVelState<double> state1;

  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  static const Eigen::Vector3d G(0, 0, -9.81);
  imu_meas.predictState(state0, G, state1);
}

// initialize the ba and bg
// (scale can be achieved from binocular, gravity is considered to be -9.81(z),
// velocity is considered to be 0)
void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                const Eigen::Vector3d& vel_w_i,
                std::vector<basalt::ImuData<double>> imu_measurements,
                const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
  basalt::ImuData<double> data = imu_measurements.front();
  while (data.t_ns < curr_frame.t_ns) {
    // data = popFromImuDataQueue();
    if (!data) break;
    data.accel = calib_cam.calib_accel_bias.getCalibrated(data.accel);
    data.gyro = calib_cam.calib_gyro_bias.getCalibrated(data.gyro);
    // std::cout << "Skipping IMU data.." << std::endl;
  }

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  Vec3 vel_w_i_init;
  vel_w_i_init.setZero();

  Sophus::SE3d T_w_i_init = T_w_i;
  T_w_i_init.setQuaternion(
      Eigen::Quaternion<double>::FromTwoVectors(data.accel, Vec3::UnitZ()));

  int64_t last_state_t_ns = curr_frame->t_ns;
  IMU_MEAS imu_meas;
  imu_meas[last_state_t_ns] =
      IntegratedImuMeasurement<double>(last_state_t_ns, bg, ba);
  FRAME_STATE frame_states;
  frame_states[last_state_t_ns] = basalt::PoseVelBiasState<double>(
      last_state_t_ns, T_w_i_init, vel_w_i_init, bg, ba, true);

  // marg_data.order.abs_order_map[last_state_t_ns] =
  //     std::make_pair(0, POSE_VEL_BIAS_SIZE);
  // marg_data.order.total_size = POSE_VEL_BIAS_SIZE;
  // marg_data.order.items = 1;

  std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
  std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
  std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;
}

}  // namespace visnav
