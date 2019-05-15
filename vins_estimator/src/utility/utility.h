#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>

class Utility
{
  public:
    // INPUT: rotate angle in axis x,y,z
    // OUTPUT: quaternion (not normalized)
    // LIMITATION: theta -> 0, sin(theta) -> theta, cos(theta) -> 1
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar T;

        Eigen::Quaternion<T> dq;
        Eigen::Matrix<T, 3, 1> half_theta = theta;
        half_theta /= T(2);
        dq.w() = T(1);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        typedef typename Derived::Scalar T;
        Eigen::Matrix<T, 3, 3> ans;
        ans << T(0), -q(2), q(1),
            q(2), T(0), -q(0),
            -q(1), q(0), T(0);
        return ans;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        typedef typename Derived::Scalar T;
        Eigen::Quaternion<T> qq = positify(q);
        Eigen::Matrix<T, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<T, 3, 3>::Identity() + skewSymmetric(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        typedef typename Derived::Scalar T;
        Eigen::Quaternion<T> pp = positify(p);
        Eigen::Matrix<T, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<T, 3, 3>::Identity() - skewSymmetric(pp.vec());
        return ans;
    }

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));                                                    // [-180, 180]
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));                          // [-90, 90]
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y)); // [-180, 180]
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g);

    template <size_t N>
    struct uint_
    {
    };

    template <size_t N, typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<N>)
    {
        unroller(f, iter, uint_<N - 1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<0>)
    {
        f(iter);
    }

    template <typename T>
    static T normalizeAngle(const T& angle_degrees) {
      T two_pi(2.0 * 180);
      if (angle_degrees > 0)
      return angle_degrees -
          two_pi * std::floor((angle_degrees + T(180)) / two_pi);
      else
        return angle_degrees +
            two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
    };

    template <typename Derived>
    static Eigen::Quaternion<Derived> meanQ(const std::vector<Eigen::Quaternion<Derived>>& Qs, const std::vector<double>& weights={}) {
        // by Tolga Birdal
        // Q is an Mx4 matrix of quaternions. weights is an Mx1 vector, a weight for
        // each quaternion.
        // Qavg is the weightedaverage quaternion
        // This function is especially useful for example when clustering poses
        // after a matching process. In such cases a form of weighting per rotation
        // is available (e.g. number of votes), which can guide the trust towards a
        // specific pose. weights might then be interpreted as the vector of votes
        // per pose.
        // Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
        // "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
        // no. 4 (2007): 1193-1197.
        // function [Qavg]=quatWAvgMarkley(Q, weights)
        //
        // % Form the symmetric accumulator matrix
        // A=zeros(4,4);
        // M=size(Q,1);
        // wSum = 0;
        //
        // for i=1:M
        //     q = Q(i,:)';
        //     w_i = weights(i);
        //     A=w_i.*(q*q')+A; % rank 1 update
        //     wSum = wSum + w_i;
        // end
        //
        // % scale
        // A=(1.0/wSum)*A;
        //
        // % Get the eigenvector corresponding to largest eigen value
        // [Qavg, ~]=eigs(A,1);
        //
        // end
        Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
        double sum_weight = 0;
        for (int i = 0; i < Qs.size(); i++) {
            Eigen::Vector4d q_i(Qs[i].x(), Qs[i].y(), Qs[i].z(), Qs[i].w());
            double w_i;
            if (weights.size() != A.size()) {
                w_i = 1.0;
            } else {
                w_i = weights[i];
            }
            A += w_i * (q_i * q_i.transpose());
            sum_weight += w_i;
        }
        A /= sum_weight;

        Eigen::EigenSolver<Eigen::Matrix4d> es(A);
        double max_eigenvalue = std::numeric_limits<double>::min();
        Eigen::Vector4d eigenvector;
        for (int i = 0; i < 4; i++) {
            std::complex<double> eigenvalue = es.eigenvalues()[i];
            if (std::norm(eigenvalue) > max_eigenvalue) {
                max_eigenvalue = std::norm(eigenvalue);
                eigenvector[0] = es.eigenvectors().col(i)[0].real();
                eigenvector[1] = es.eigenvectors().col(i)[1].real();
                eigenvector[2] = es.eigenvectors().col(i)[2].real();
                eigenvector[3] = es.eigenvectors().col(i)[3].real();
            }
        }

        return Eigen::Quaternion<Derived>(eigenvector[3], eigenvector[0], eigenvector[1], eigenvector[2]);
    }

    static double meanAngle(std::vector<double> angles, std::vector<double> weights = {}, double range = 2 * M_PI) {
        if (angles.empty())
            throw std::invalid_argument("vector of angles should not be empty");
        if (weights.size() != angles.size() && !weights.empty()) {
            throw std::invalid_argument("vector of weights should be empty or has same size with vector of angles");
        }

        double sum_angle = 0;
        double sum_weight = 0;
        for (int i = 0; i < angles.size(); i++) {
            double& angle = angles[i];
            double weight;
            if (i < weights.size())
                weight = weights[i];
            else
                weight = 1;

            if (sum_weight != 0) {
                // divide angle to [sum_angle - pi, sum_angle + pi]
                angle -= sum_angle / sum_weight;
                angle = std::remainder(angle, range);
                angle += sum_angle / sum_weight;
            }

            sum_angle += weight * angle;
            sum_weight += weight;
        }

        double mean_angle = std::remainder(sum_angle / sum_weight, range);
        return mean_angle;
    }

    // https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    static double weightedStd(std::vector<double> diffs, std::vector<double> weights = {}) {
        if (diffs.empty())
            throw std::invalid_argument("vector of diffs should not be empty");
        if (weights.size() != diffs.size() && !weights.empty()) {
            throw std::invalid_argument("vector of weights should be empty or has same size with vector of diffs");
        }

        double sum_weighted_var = 0;
        double sum_weight = 0;
        double sum_squared_weight = 0;
        for (int i = 0; i < diffs.size(); i++) {
            double& diff = diffs[i];
            double weight;
            if (i < weights.size())
                weight = weights[i];
            else
                weight = 1;
            sum_weighted_var += weight * std::pow(diff, 2);
            sum_weight += weight;
            sum_squared_weight += std::pow(weight, 2);
        }
//    double std = std::sqrt(sum_weighted_var / (sum_weight - sum_squared_weight / sum_weight));
        double std = std::sqrt(sum_weighted_var / sum_weight);
        return std;
    }
};
