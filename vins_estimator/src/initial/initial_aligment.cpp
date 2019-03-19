#include <fstream>

#include "../utility/tic_toc.h"

#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

std::vector<double> GetStillFrames(const map<double, ImageFrame> &all_image_frame, int protect_frames) {
    std::vector<pair<bool, double>> durations;
    durations.emplace_back(std::make_pair(false, 0));
    bool first = true;
    for (const auto& it : all_image_frame) {
        if (first) {
            first = false;
            continue;
        }
        if (it.second.base_integration->still) {
            durations.emplace_back(std::make_pair(true, it.second.base_integration->sum_dt));
//            std::cout << 'T';
        } else {
            durations.emplace_back(std::make_pair(false, 0));
//            std::cout << 'F';
        }
    }
//    std::cout << std::endl;
    durations.emplace_back(std::make_pair(false, 0));

    int protection_frame_count = 1;
    for (int i = 1; i < durations.size() - 1; i++) {
        if (!durations[i].first)
            continue;
        for (int j = i - protection_frame_count; j <= i + protection_frame_count; j++) {
            if (j >= 0 && j < durations.size() && !durations[j].first) {
                durations[i].second = 0;
                break;
            }
        }
    }

    int i = 1;
    std::vector<double> still_frames;
    for (const auto& kv : all_image_frame) {
        if (durations[i].second > 0) {
            still_frames.push_back(kv.first);
        }
        i++;
    }

    return still_frames;
}

Vector3d solveStillGyroBias(map<double, ImageFrame> &all_image_frame) {

    std::ofstream ofs("/tmp/solveStillGyroBias.csv");

    Vector3d sum_gyro = Vector3d::Zero();
    double sum_dt = 0;
    for (double k : GetStillFrames(all_image_frame, 1)) {
        for (int i = 0; i < all_image_frame[k].pre_integration->dt_buf.size(); i++) {
            sum_dt += all_image_frame[k].pre_integration->dt_buf[i];
            sum_gyro += all_image_frame[k].pre_integration->gyr_buf[i] * all_image_frame[k].pre_integration->dt_buf[i];
            ofs << all_image_frame[k].pre_integration->dt_buf[i] << ','
                << all_image_frame[k].pre_integration->gyr_buf[i].x() << ','
                << all_image_frame[k].pre_integration->gyr_buf[i].y() << ','
                << all_image_frame[k].pre_integration->gyr_buf[i].z() << std::endl;
        }
    }
    ofs.close();

    Vector3d bias_gyro = sum_gyro / sum_dt;
    return bias_gyro;
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

// g [INPUT & OUTPUT]: gravity in cam0 frame
// x [OUTPUT]: result vector
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // iterate to refine gravity
    for(int k = 0; k < 4; k++)
    {
        // calc b_1, b_2 as paper algorithm 1.
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);

        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            // gravity is 2Dims
            // 3+3+2+1=9 Dims
            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        //double s = x(n_state - 1);
    }

    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double err = (A * x - b).norm();
    ROS_INFO("error: %f", err);
    double s = x(n_state - 1) / 100.0;
    ROS_INFO("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_INFO_STREAM(" result g     " << g.norm() << " " << (RIO.transpose() * RIC[0] * g).transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_INFO_STREAM(" refine     " << g.norm() << " " << (RIO.transpose() * RIC[0] * g).transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// g [INPUT & OUTPUT]: gravity in IMU frame
// x [OUTPUT]: result vector
void BaseIMURefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
}

void base_imu_alignment(const vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> &pre_integrations,
                        const Matrix3d& R_imu_base, const Vector3d& t_imu_base,
                        VectorXd &x, Vector3d &g, double &s, double &avg_err_p, double &avg_err_v) {

    int frame_count = pre_integrations.size();
    int n_state = frame_count * 3 + 3 + 1;

    MatrixXd AA{n_state, n_state};              AA.setZero();
    VectorXd Ab{n_state};                       Ab.setZero();

    Matrix3d R_b0_bi = Matrix3d::Identity();
    for (int i = 0; i < frame_count - 1; i++) {
        auto& frame_i = pre_integrations[i];
        auto& frame_j = pre_integrations[i+1];
        Matrix3d R_bi_bj = frame_j.first->delta_q.toRotationMatrix();

        MatrixXd tmp_A(6, 10);  tmp_A.setZero();
        VectorXd tmp_b(6);      tmp_b.setZero();

        double dt = frame_j.first->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = (dt * dt / 2) * R_b0_bi.transpose();
        tmp_A.block<3, 1>(0, 9) = R_imu_base * frame_j.second->delta_p / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j.first->delta_p + R_bi_bj * t_imu_base - t_imu_base;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_bi_bj;
        tmp_A.block<3, 3>(3, 6) = R_b0_bi.transpose() * dt;
        tmp_b.block<3, 1>(3, 0) = frame_j.first->delta_v;

        MatrixXd r_A = tmp_A.transpose() * tmp_A;
        VectorXd r_b = tmp_A.transpose() * tmp_b;

        AA.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        Ab.segment<6>(i * 3) += r_b.head<6>();

        AA.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        Ab.tail<4>() += r_b.tail<4>();

        AA.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        AA.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();

        R_b0_bi = R_b0_bi * R_bi_bj;
    }

    AA = AA * 1000.0;
    Ab = Ab * 1000.0;
    x = AA.ldlt().solve(Ab);

    // validate error
    avg_err_p = 0;
    avg_err_v = 0;
    R_b0_bi = Matrix3d::Identity();
    double sum_distance = 0;
    double sum_dt = 0;
    for (int i = 0; i < frame_count - 1; i++) {
        auto& frame_i = pre_integrations[i];
        auto& frame_j = pre_integrations[i+1];
        Matrix3d R_bi_bj = frame_i.first->delta_q.toRotationMatrix();

        MatrixXd tmp_A(6, 10);  tmp_A.setZero();
        VectorXd tmp_b(6);      tmp_b.setZero();

        double dt = frame_j.first->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = (dt * dt / 2) * R_b0_bi.transpose();
        tmp_A.block<3, 1>(0, 9) = R_imu_base * frame_j.second->delta_p / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j.first->delta_p + R_bi_bj * t_imu_base - t_imu_base;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_bi_bj;
        tmp_A.block<3, 3>(3, 6) = R_b0_bi.transpose() * dt;
        tmp_b.block<3, 1>(3, 0) = frame_j.first->delta_v;

        VectorXd tmp_x(10);
        tmp_x.head(6) = x.segment<6>(i * 3);
        tmp_x.tail(4) = x.tail(4);

        VectorXd res(6);
        res = tmp_A * tmp_x - tmp_b;
        avg_err_p += res.head<3>().norm();
        avg_err_v += res.tail<3>().norm();
        sum_distance += frame_j.second->delta_p.norm();
        sum_dt += dt;

        R_b0_bi = R_b0_bi * R_bi_bj;
    }
    avg_err_p /= (frame_count - 1);
    avg_err_v /= (frame_count - 1);
    s = (x(n_state - 1) / 100.0);
    g = x.segment<3>(n_state - 4);

//    ROS_INFO("base_imu_alignment n:%d dt:%f dp:%f err_p:%f err_v:%f scale:%f g:%f", frame_count, sum_dt, sum_distance, avg_err_p, avg_err_v, s, g.norm());
//    ROS_INFO_STREAM("base_imu_alignment g " << g.norm() << " " << (R_imu_base.transpose() * g).transpose());
}

void base_imu_alignment_fixed_scale(const vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> &pre_integrations,
                        const Matrix3d& R_imu_base, const Vector3d& t_imu_base,
                        VectorXd &x, Vector3d &g, double &avg_err_p, double &avg_err_v) {

    int frame_count = pre_integrations.size();
    int n_state = frame_count * 3 + 3;

    MatrixXd AA{n_state, n_state};              AA.setZero();
    VectorXd Ab{n_state};                       Ab.setZero();

    Matrix3d R_b0_bi = Matrix3d::Identity();
    for (int i = 0; i < frame_count - 1; i++) {
        auto& frame_i = pre_integrations[i];
        auto& frame_j = pre_integrations[i+1];
        Matrix3d R_bi_bj = frame_j.first->delta_q.toRotationMatrix();

        MatrixXd tmp_A(6, 9);  tmp_A.setZero();
        VectorXd tmp_b(6);      tmp_b.setZero();

        double dt = frame_j.first->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = (dt * dt / 2) * R_b0_bi.transpose();
        tmp_b.block<3, 1>(0, 0) = frame_j.first->delta_p + R_bi_bj * t_imu_base - t_imu_base - R_imu_base * frame_j.second->delta_p;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_bi_bj;
        tmp_A.block<3, 3>(3, 6) = R_b0_bi.transpose() * dt;
        tmp_b.block<3, 1>(3, 0) = frame_j.first->delta_v;

        MatrixXd r_A = tmp_A.transpose() * tmp_A;
        VectorXd r_b = tmp_A.transpose() * tmp_b;

        AA.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        Ab.segment<6>(i * 3) += r_b.head<6>();

        AA.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        Ab.tail<3>() += r_b.tail<3>();

        AA.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        AA.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();

        R_b0_bi = R_b0_bi * R_bi_bj;
    }

    AA = AA * 1000.0;
    Ab = Ab * 1000.0;
    x = AA.ldlt().solve(Ab);

    // validate error
    avg_err_p = 0;
    avg_err_v = 0;
    R_b0_bi = Matrix3d::Identity();
    double sum_distance = 0;
    double sum_dt = 0;
    for (int i = 0; i < frame_count - 1; i++) {
        auto& frame_i = pre_integrations[i];
        auto& frame_j = pre_integrations[i+1];
        Matrix3d R_bi_bj = frame_i.first->delta_q.toRotationMatrix();

        MatrixXd tmp_A(6, 9);  tmp_A.setZero();
        VectorXd tmp_b(6);      tmp_b.setZero();

        double dt = frame_j.first->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = (dt * dt / 2) * R_b0_bi.transpose();
        tmp_b.block<3, 1>(0, 0) = frame_j.first->delta_p + R_bi_bj * t_imu_base - t_imu_base - R_imu_base * frame_j.second->delta_p;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_bi_bj;
        tmp_A.block<3, 3>(3, 6) = R_b0_bi.transpose() * dt;
        tmp_b.block<3, 1>(3, 0) = frame_j.first->delta_v;

        VectorXd tmp_x(9);
        tmp_x.head(6) = x.segment<6>(i * 3);
        tmp_x.tail(3) = x.tail(3);

        VectorXd res(6);
        res = tmp_A * tmp_x - tmp_b;
        avg_err_p += res.head<3>().norm();
        avg_err_v += res.tail<3>().norm();
        sum_distance += frame_j.second->delta_p.norm();
        sum_dt += dt;

        R_b0_bi = R_b0_bi * R_bi_bj;
    }
    avg_err_p /= (frame_count - 1);
    avg_err_v /= (frame_count - 1);
    g = x.segment<3>(n_state - 4);

//    ROS_INFO("base_imu_alignment_fixed_scale dt:%f dp:%f err_p:%f err_v:%f g:%f", sum_dt, sum_distance, avg_err_p, avg_err_v, g.norm());
//    ROS_INFO_STREAM("base_imu_alignment g " << g.norm() << " " << (R_imu_base.transpose() * g).transpose());
}

// base wheel odometry align with IMU
bool BaseIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x, VectorXd &refined_x) {

    TicToc tic;

    vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> frames;
    for (const auto& kv : all_image_frame) {
        const auto& bg = kv.second.pre_integration->linearized_bg;
        kv.second.base_integration->repropagate(Eigen::Matrix3d::Identity(), bg);
        frames.emplace_back(make_pair(kv.second.pre_integration, kv.second.base_integration));
    }
    double s;
    double avg_err_p;
    double avg_err_v;
    base_imu_alignment(frames, RIO, TIO, x, g, s, avg_err_p, avg_err_v);

    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
        return false;
    }

    ROS_INFO("cost %f ms", tic.toc());
    tic.tic();

//    RefineGravity(all_image_frame, g, x);
//    s = (x.tail<1>())(0) / 100.0;
//    (x.tail<1>())(0) = s;
//    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
//    if(s < 0.0 )
//        return false;
//    else
//        return true;

    {

        // R_b0_bk
        std::vector<Eigen::Matrix3d> rotations;
        double wheel_distance = 0;
        for (auto it = all_image_frame.begin(); it != all_image_frame.end(); it++) {
            const auto& bg = it->second.pre_integration->linearized_bg;
            it->second.base_integration->repropagate(Eigen::Matrix3d::Identity(), bg);
            if (rotations.empty()) {
                rotations.push_back(Eigen::Matrix3d::Identity());
            } else {
                rotations.emplace_back(it->second.pre_integration->delta_q.toRotationMatrix() * rotations.back());
            }
            wheel_distance += it->second.base_integration->delta_p.norm();
        }
        ROS_INFO("wheel_distance: %f", wheel_distance);

        Vector3d g0 = g.normalized() * G.norm();
        Vector3d lx, ly;
        //VectorXd x;
        int all_frame_count = all_image_frame.size();
        int n_state = all_frame_count * 3 + 2 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();

        map<double, ImageFrame>::iterator frame_i;
        map<double, ImageFrame>::iterator frame_j;
        // iterate to refine gravity
        for(int k = 0; k < 4; k++)
        {
            // calc b_1, b_2 as paper algorithm 1.
            MatrixXd lxly(3, 2);
            lxly = TangentBasis(g0);

            int i = 0;
            for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
            {
                frame_j = next(frame_i);
                const auto& R_b0_bi = rotations[i];
                const auto& R_b0_bj = rotations[i+1];
                auto R_bi_bj = R_b0_bi.transpose() * R_b0_bj;

                // gravity is 2Dims
                // 3+3+2+1=9 Dims
                MatrixXd tmp_A(6, 9);
                tmp_A.setZero();
                VectorXd tmp_b(6);
                tmp_b.setZero();

                double dt = frame_j->second.pre_integration->sum_dt;

                tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
                tmp_A.block<3, 2>(0, 6) = (dt * dt / 2) * R_b0_bi.transpose() * lxly;
                tmp_A.block<3, 1>(0, 8) = RIO * frame_j->second.base_integration->delta_p / 100.0;
                tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + R_bi_bj * TIO - TIO - (dt * dt / 2) * R_b0_bi.transpose() * g0;

                tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
                tmp_A.block<3, 3>(3, 3) = R_bi_bj;
                tmp_A.block<3, 2>(3, 6) = R_b0_bi.transpose() * dt * lxly;
                tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - R_b0_bi.transpose() * dt * g0;


                Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
                //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
                //MatrixXd cov_inv = cov.inverse();
                cov_inv.setIdentity();

                MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
                VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();

                A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            }
            A = A * 1000.0;
            b = b * 1000.0;
            refined_x = A.ldlt().solve(b);
            VectorXd dg = refined_x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            double s = refined_x(n_state - 1);
        }

        g = g0;
    }

    ROS_INFO_STREAM("wheel-odom & IMU refine g: " << g.norm() << " " << (RIO.transpose() * g).transpose());
    ROS_INFO("cost %f ms", tic.toc());

    return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    auto still_bg = solveStillGyroBias(all_image_frame);
    ROS_INFO_STREAM("Bg VIO:" << Bgs[WINDOW_SIZE].transpose() << " Still:" << still_bg.transpose());

    VectorXd refined_x;
    BaseIMUAlignment(all_image_frame, g, x, refined_x);

    auto x_backup = x;

    if(LinearAlignment(all_image_frame, g, x)) {
//        for (int i = 0; i < all_image_frame.size(); i++) {
//            Matrix3d m;
//            m.col(0) = RIO.transpose() * x.segment<3>(i * 3);
//            m.col(1) = RIO.transpose() * refined_x.segment<3>(i * 3);
//            m.col(2) = RIO.transpose() * x_backup.segment<3>(i * 3);
//            ROS_INFO_STREAM(i << " v^base_imu:" << std::endl << m);
//        }
        return true;
    } else
        return false;
}
