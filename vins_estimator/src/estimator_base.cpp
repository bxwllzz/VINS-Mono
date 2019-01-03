//
// Created by bxwllzz on 18-12-5.
//

#include "estimator.h"

bool Estimator::baseOdomAlign() {
    // calc mean pitch and roll angle of R^base_imu at initializing
    std::vector<Eigen::Quaterniond> rots;
    std::vector<Eigen::Vector3d> yprs;
    for (int i = 0; i <= frame_count; i++) {
        rots.emplace_back(Rs[i]);
        yprs.emplace_back(Utility::R2ypr(Rs[i]) / 180 * M_PI);
    }
    Eigen::Quaterniond mean_q = Utility::meanQ(rots);
    Eigen::Vector3d mean_ypr = Utility::R2ypr(mean_q.toRotationMatrix()) / 180 * M_PI;
    double mean_pitch = mean_ypr[1];
    double mean_roll = mean_ypr[2];
    // std of pitch & roll
    double sum_pitch_var = 0;
    double sum_roll_var = 0;
    for (int i = 0; i <= frame_count; i++) {
        sum_pitch_var += std::pow(std::remainder(yprs[i][1] - mean_pitch, 2 * M_PI), 2);
        sum_roll_var += std::pow(std::remainder(yprs[i][2] - mean_roll, 2 * M_PI), 2);
    }
    double std_pitch = std::sqrt(sum_pitch_var / (frame_count + 1));
    double std_roll = std::sqrt(sum_roll_var / (frame_count + 1));

    // guese yaw angle of R^base_imu
    std::vector<double> yaws;
    std::vector<double> weights;
    for (int i = 0; i < frame_count; i++) {
        int j = i + 1;

        Eigen::Affine2d T_i0 = Eigen::Translation2d(Ps[i][0], Ps[i][1]) * Eigen::Rotation2Dd(Utility::R2ypr(Rs[i])[0] / 180 * M_PI);
        Eigen::Affine2d T_i1 = Eigen::Translation2d(Ps[j][0], Ps[j][1]) * Eigen::Rotation2Dd(Utility::R2ypr(Rs[j])[0] / 180 * M_PI);
        Eigen::Affine2d T_i0_i1 = T_i0.inverse() * T_i1;
        Eigen::Vector2d dp_vio = T_i0_i1.translation();

        Eigen::Vector2d dp_bo = base_integrations[j]->delta_p;

        double dir_vio = atan2(dp_vio[1], dp_vio[0]);
        double dir_bo = atan2(dp_bo[1], dp_bo[0]);
        double min_dist = std::min(dp_vio.norm(), dp_bo.norm());

        double yaw = std::remainder(dir_bo - dir_vio, 2 * M_PI);
        double weight = min_dist;
        yaws.push_back(yaw);
        weights.push_back(weight);
        ROS_INFO_STREAM("dir_vio=" << dir_vio / M_PI * 180 << " dir_bo=" << dir_bo / M_PI * 180 << " yaw=" << yaw / M_PI * 180 << " weight=" << weight);
    }
    // weighted mean yaw
    double mean_yaw = Utility::meanAngle(yaws, weights);

    // weighted std
    for (auto& yaw : yaws)
        yaw = std::remainder(yaw - mean_yaw, 2 * M_PI);
    double std_yaw = Utility::weightedStd(yaws, weights);

    ROS_WARN("Initialize R^base_imu, yaw=%.3f(std %.3f), pitch=%.3f(std %.3f), roll=%.3f(std %.3f)",
             mean_yaw / M_PI * 180, std_yaw / M_PI * 180,
             mean_pitch / M_PI * 180, std_pitch / M_PI * 180,
             mean_roll / M_PI * 180, std_roll / M_PI * 180);

    Eigen::Affine3d Tib = Eigen::Translation3d(tib) * rib;
    Eigen::Affine3d Tbi = Tib.inverse();
    Tbi.linear() = Utility::ypr2R(Eigen::Vector3d(mean_yaw / M_PI * 180, mean_pitch / M_PI * 180, mean_roll / M_PI * 180));
    Tib = Tbi.inverse();
    rib = Tib.rotation();
    tib = Tib.translation();

    // save initialization result
//    std::ofstream ofs("/home/bxwllzz/init_result.csv");
//    ofs << "Px,Py,Pz,Yaw,Pitch,Roll,Vx,Vy,Vz,Ix_BO,Iy_BO,Iw_BO,Px_BO,Py_BO,Pw_BO" << std::endl;
//    BaseOdometryIntegration base_integration({0, 0}, 0);
//    for (int i = 0; i <= frame_count; i++) {
//        ofs << Ps[i].x() << ',' << Ps[i].y() << ',' << Ps[i].z() << ',';
//        Eigen::Vector3d ypr = Utility::R2ypr(Rs[i]) / 180 * M_PI;
//        ofs << ypr[0] << ',' << ypr[1] << ',' << ypr[2] << ',';
//        ofs << Vs[i].x() << ',' << Vs[i].y() << ',' << Vs[i].z() << ',';
//        ofs << base_integrations[i]->delta_p.x() << ',' << base_integrations[i]->delta_p.y() << ',' << base_integrations[i]->delta_yaw << ',';
//        for (auto& m : base_integrations[i]->measurements) {
//            base_integration.propagate(m);
//        }
//        ofs << base_integration.delta_p.x() << ',' << base_integration.delta_p.y() << ',' << base_integration.delta_yaw << std::endl;
//    }
//    ROS_WARN_STREAM("Initialization base result saved");

    return true;
}