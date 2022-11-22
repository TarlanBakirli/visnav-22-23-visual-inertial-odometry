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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  T t = xi.norm();

  if (t == T(0)) {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -xi(2), xi(1), xi(2), 0, -xi(0), -xi(1), xi(0), 0;

  return Eigen::Matrix<T, 3, 3>::Identity() + sin(t) / t * w_hat +
         (T(1) - cos(t)) / pow(t, T(2)) * (w_hat * w_hat);
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  T t = acos((mat.trace() - T(1)) / T(2));

  if (t == T(0)) {
    return Eigen::Matrix<T, 3, 1>::Zero();
  }

  Eigen::Matrix<T, 3, 1> r;
  r << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);

  Eigen::Matrix<T, 3, 1> v = (T(1) / (T(2) * sin(t))) * r;
  UNUSED(mat);
  return t * v;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> v = xi.head(3);
  Eigen::Matrix<T, 3, 1> w = xi.tail(3);

  T t = w.norm();

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  Eigen::Matrix<T, 3, 3> J;
  if (t == T(0)) {
    J = Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    J = Eigen::Matrix<T, 3, 3>::Identity() +
        (T(1) - cos(t)) / pow(t, T(2)) * w_hat +
        (t - sin(t)) / pow(t, T(3)) * (w_hat * w_hat);
  }

  Eigen::Matrix<T, 3, 3> SO3 = user_implemented_expmap(w);

  Eigen::Matrix<T, 4, 4> result;
  result << SO3, J * v, Eigen::Matrix<T, 1, 3>::Zero(), T(1);
  UNUSED(xi);
  return result;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);

  T theta = acos((R.trace() - T(1)) / T(2));

  Eigen::Matrix<T, 3, 1> w;
  Eigen::Matrix<T, 3, 3> J_inv;

  if (theta == T(0)) {
    w = Eigen::Matrix<T, 3, 1>::Zero();
    J_inv = Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    w = user_implemented_logmap(R);
    Eigen::Matrix<T, 3, 3> w_hat;
    w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    J_inv = Eigen::Matrix<T, 3, 3>::Identity() - T(1) / T(2) * w_hat +
            (T(1) / pow(theta, 2) -
             (T(1) + cos(theta)) / (T(2) * theta * sin(theta))) *
                (w_hat * w_hat);
  }
  Eigen::Matrix<T, 6, 1> result;
  result << J_inv * t, w;
  UNUSED(mat);
  return result;  // Eigen::Matrix<T, 6, 1>();
}

}  // namespace visnav
