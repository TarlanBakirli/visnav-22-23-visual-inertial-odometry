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
void imu_preintegration(const FrameId fid) {
  // input: vector<ImuData>, vector<timestamps>, frameId, frame_state
  // output: save in frame_state/frame_pose

  // When we use imu_meas?

  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  basalt::PoseVelState<double> state0;
  basalt::PoseVelState<double> state1;

  // imu_meas;

  // state0.T_w_i = gt_spline.pose(int64_t(0));
  // state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(20e9);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    basalt::ImuData<double> data;
    data.accel = accel_body;
    data.gyro = rot_vel_body;
    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  imu_meas.predictState(state0, basalt::constants::G, state1);
}

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

}  // namespace visnav