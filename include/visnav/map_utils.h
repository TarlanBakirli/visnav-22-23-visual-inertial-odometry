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

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map
  opengv::bearingVectors_t bearingVectors1, bearingVectors2;

  for (const auto& track_id : shared_track_ids) {
    if (landmarks.find(track_id) == landmarks.end()) {
      new_track_ids.push_back(track_id);
      const KeypointsData kd1 = feature_corners.at(fcid0);
      const KeypointsData kd2 = feature_corners.at(fcid1);
      const Eigen::Vector2d p0_2d =
          kd1.corners[feature_tracks.at(track_id).at(fcid0)];
      const Eigen::Vector2d p1_2d =
          kd2.corners[feature_tracks.at(track_id).at(fcid1)];
      const std::shared_ptr<AbstractCamera<double>>& cam1 =
          calib_cam.intrinsics[fcid0.cam_id];
      const std::shared_ptr<AbstractCamera<double>>& cam2 =
          calib_cam.intrinsics[fcid1.cam_id];
      bearingVectors1.push_back(cam1->unproject(p0_2d).normalized());
      bearingVectors2.push_back(cam2->unproject(p1_2d).normalized());
    }
  }

  Sophus::SE3d transformation =
      cameras.at(fcid0).T_w_c.inverse() * cameras.at(fcid1).T_w_c;

  Eigen::Vector3d translation = transformation.translation();
  Eigen::Matrix3d rotation = transformation.rotationMatrix();

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors1, bearingVectors2, translation, rotation);

  size_t index = 0;
  for (const auto& track_id : new_track_ids) {
    // run method 1
    opengv::point_t point = opengv::triangulation::triangulate(adapter, index);
    landmarks[track_id].p = cameras.at(fcid0).T_w_c * point;
    for (const auto& feature_track : feature_tracks.at(track_id))
      if (cameras.find(feature_track.first) != cameras.end())
        landmarks[track_id].obs.emplace(feature_track);
    index++;
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)

  cameras[fcid0].T_w_c = calib_cam.T_i_c[fcid0.cam_id];
  cameras[fcid1].T_w_c = calib_cam.T_i_c[fcid1.cam_id];

  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map

  const std::shared_ptr<AbstractCamera<double>>& cam =
      calib_cam.intrinsics[fcid.cam_id];

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (const auto& track_id : shared_track_ids) {
    const KeypointsData kd = feature_corners.at(fcid);
    const Eigen::Vector2d p_2d =
        kd.corners[feature_tracks.at(track_id).at(fcid)];
    bearingVectors.push_back(cam->unproject(p_2d).normalized());
    points.push_back(landmarks.at(track_id).p);
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
    inlier_track_ids.push_back(shared_track_ids[inlier]);
  T_w_c = Sophus::SE3d(nonlinear_transformation.topLeftCorner(3, 3),
                       nonlinear_transformation.topRightCorner(3, 1));
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  // Define loss function
  ceres::HuberLoss* loss_function =
      options.use_huber ? new ceres::HuberLoss(options.huber_parameter)
                        : nullptr;

  // Add camera intrinsics
  problem.AddParameterBlock(calib_cam.intrinsics[0]->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics[1]->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  }

  // Add camera extrinsics
  for (auto& camera : cameras) {
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    if (fixed_cameras.find(camera.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());
    }
  }

  // Add landmarks and residual block
  for (auto& landmark : landmarks) {
    problem.AddParameterBlock(landmark.second.p.data(), 3);
    // Add observations
    for (const auto& obs : landmark.second.obs) {
      Eigen::Vector2d p_2d =
          feature_corners.at(obs.first).corners.at(obs.second);

      // Create ceres cost function
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2, 7, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(
              p_2d, calib_cam.intrinsics[obs.first.cam_id]->name()));

      problem.AddResidualBlock(cost_function, loss_function,
                               cameras[obs.first].T_w_c.data(),
                               landmark.second.p.data(),
                               calib_cam.intrinsics[obs.first.cam_id]->data());
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment_without_IMU(const Corners& feature_corners,
                                   const BundleAdjustmentOptions& options,
                                   const std::set<FrameCamId>& fixed_cameras,
                                   Calibration& calib_cam, Cameras& cameras,
                                   Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  // Define loss function
  ceres::HuberLoss* loss_function =
      options.use_huber ? new ceres::HuberLoss(options.huber_parameter)
                        : nullptr;

  // Add camera intrinsics
  problem.AddParameterBlock(calib_cam.intrinsics[0]->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics[1]->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  }

  // Add camera extrinsics
  for (auto& camera : cameras) {
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    if (fixed_cameras.find(camera.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());
    }
  }

  // Add landmarks and residual block
  for (auto& landmark : landmarks) {
    problem.AddParameterBlock(landmark.second.p.data(), 3);
    // Add observations
    for (const auto& obs : landmark.second.obs) {
      Eigen::Vector2d p_2d =
          feature_corners.at(obs.first).corners.at(obs.second);

      // Create ceres cost function
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2, 7, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(
              p_2d, calib_cam.intrinsics[obs.first.cam_id]->name()));

      problem.AddResidualBlock(cost_function, loss_function,
                               cameras[obs.first].T_w_c.data(),
                               landmark.second.p.data(),
                               calib_cam.intrinsics[obs.first.cam_id]->data());
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

// Run bundle adjustment to optimize cameras, points using frames and IMU
void bundle_adjustment_with_IMU(
    const Corners& feature_corners, const BundleAdjustmentOptions& options,
    const std::set<FrameCamId>& fixed_cameras, Calibration& calib_cam,
    Cameras& cameras, Landmarks& landmarks, IMU_MEAS& imu_meas,
    FRAME_STATE& frame_states, std::set<FrameId> kf_frames,
    std::set<FrameId> buffer_frames, std::vector<Timestamp> timestamps) {
  // , IMUs& imus
  ceres::Problem problem;

  // Setup optimization problem
  // Define loss function
  ceres::HuberLoss* loss_function =
      options.use_huber ? new ceres::HuberLoss(options.huber_parameter)
                        : nullptr;

  // For camera part:
  // Add camera intrinsics
  problem.AddParameterBlock(calib_cam.intrinsics[0]->data(), 8);
  problem.AddParameterBlock(calib_cam.intrinsics[1]->data(), 8);
  if (!options.optimize_intrinsics) {
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  }

  // Add camera extrinsics
  for (auto& camera : cameras) {
    problem.AddParameterBlock(camera.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);

    if (fixed_cameras.find(camera.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(camera.second.T_w_c.data());
    }
  }

  // Add landmarks and residual block
  for (auto& landmark : landmarks) {
    problem.AddParameterBlock(landmark.second.p.data(), 3);
    // Add observations
    for (const auto& obs : landmark.second.obs) {
      Eigen::Vector2d p_2d =
          feature_corners.at(obs.first).corners.at(obs.second);

      // Create ceres cost function
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2, 7, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(
              p_2d, calib_cam.intrinsics[obs.first.cam_id]->name()));

      problem.AddResidualBlock(cost_function, loss_function,
                               cameras[obs.first].T_w_c.data(),
                               landmark.second.p.data(),
                               calib_cam.intrinsics[obs.first.cam_id]->data());
    }
  }

  // For IMU part:
  // Add residual block for IMU
  for (const auto& buffer_frame : buffer_frames) {
    size_t buffer_t_ns = timestamps[buffer_frame];
    problem.AddParameterBlock(frame_states[buffer_t_ns].T_w_i.data(), 7,
                              new Sophus::test::LocalParameterizationSE3);
    problem.AddParameterBlock(frame_states[buffer_t_ns].vel_w_i.data(), 3);
    if (buffer_frame == *buffer_frames.begin())
      problem.SetParameterBlockConstant(
          frame_states[buffer_t_ns].vel_w_i.data());
  }

  // Create ceres cost function
  if (buffer_frames.size() > 1) {
    Eigen::Vector3d curr_bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d curr_ba = Eigen::Vector3d::Zero();

    for (auto buffer_frame : buffer_frames) {
      size_t curr_t_ns = timestamps[buffer_frame];
      size_t last_t_ns = timestamps[buffer_frame - 1];
      ceres::CostFunction* cost_function =
          new ceres::NumericDiffCostFunction<BundleAdjustmentIMUCostFunctor,
                                             ceres::CENTRAL, 9, 7, 7, 3, 3>(
              new BundleAdjustmentIMUCostFunctor(imu_meas[curr_t_ns], curr_bg,
                                                 curr_ba));
      problem.AddResidualBlock(cost_function, loss_function,
                               frame_states[last_t_ns].T_w_i.data(),
                               frame_states[curr_t_ns].T_w_i.data(),
                               frame_states[last_t_ns].vel_w_i.data(),
                               frame_states[curr_t_ns].vel_w_i.data());
    }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
