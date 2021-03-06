#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/base_odom_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/plane_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

#include "utility/ImuUtility.h"
#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;

    rio = RIO;
    tio = TIO;
    td_bo = TD_ODOM;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();

        pre_integrations[i].reset();
        base_integrations[i].reset();
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    solver_flag = INITIAL;
    first_imu = true,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    rio = Matrix3d::Identity();
    tio = Vector3d::Zero();
    td_bo = TD_ODOM;

    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration.reset();
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();

    base_integration_before_init = BaseOdometryIntegration3D();
    wheel_only_odom = BaseOdometryIntegration3D();
    wheel_imu_odom = BaseOdometryIntegration3D();
    wheel_imu_odom3D = BaseOdometryIntegration3D();
    tmp_base_integration.reset();
    wheel_imu_predict.reset();
    wheel_slip_periods.clear();
    slipping = false;

    status.clear();
    history_index.clear();
    history_status.clear();
    prev_position = {0, 0, 0};
    total_distance = 0;
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (first_imu) {
        first_imu = false;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        return;
    }

    if (!pre_integrations[frame_count]) {
        pre_integrations[frame_count] = std::make_shared<IntegrationBase>(acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]);
    }
    if (!tmp_pre_integration) {
        tmp_pre_integration = std::make_shared<IntegrationBase>(acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]);
    }
//    if (!wi_imu_integrations.back()) {
//        wi_imu_integrations.back() = make_shared<IntegrationBase>(acc_0, gyr_0, Vector3d::Zero(), Bgs[frame_count]);
//    }
//
//    wi_imu_integrations.back()->push_back(dt, linear_acceleration, angular_velocity);
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
    //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

    // midpoint integration (only valid after intialization)
    // prepare init guese for non-linear optimization
    auto& j = frame_count;
    auto& acc_1 = linear_acceleration;
    auto& gyr_1 = angular_velocity;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // without g (g is valid after initialazation)
    Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (acc_1 - Bas[j]) - g; // without g (g is valid after initialazation)
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;

    //debug
//    wheel_imu_P += dt * wheel_imu_V + 0.5 * dt * dt * un_acc;
//    wheel_imu_V += dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processOdometry(
        double dt,
        const pair<Vector2d, double>& velocity,
        double constraint_error_vel,
        Vector3d imu_linear_acceleration,
        Vector3d imu_angular_velocity) {

    if (!tmp_base_integration)
        tmp_base_integration = std::make_shared<BaseOdometryIntegration3D>(Bgs[frame_count]);
    tmp_base_integration->push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            imu_angular_velocity,
            imu_linear_acceleration));

    if (!base_integrations[frame_count])
        base_integrations[frame_count] = std::make_shared<BaseOdometryIntegration3D>(Bgs[frame_count]);
    base_integrations[frame_count]->push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            imu_angular_velocity,
            imu_linear_acceleration));

    if (!wheel_imu_predict)
        wheel_imu_predict = make_shared<BaseOdometryIntegration3D>(Vector3d::Zero());
    // use wheelodom yaw, imu pitch roll
    Vector3d angular_velocity_odom = rio.inverse() * (imu_angular_velocity - Bgs[frame_count]);
    angular_velocity_odom.z() = velocity.second;
    wheel_imu_predict->push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            rio * angular_velocity_odom,
            imu_linear_acceleration));

    wheel_only_odom.push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            rio * Vector3d(0, 0, velocity.second),
            imu_linear_acceleration));
    Vector3d bg = solver_flag == NON_LINEAR ? Bgs[frame_count] : Vector3d::Zero();
    wheel_imu_odom.push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            rio * Vector3d(0, 0, (rio.transpose() * (imu_angular_velocity - bg)).z()),
            imu_linear_acceleration));
    wheel_imu_odom3D.push_back(MixedOdomMeasurement(dt, velocity, constraint_error_vel,
            imu_angular_velocity - bg,
            imu_linear_acceleration));
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    status.clear();

    ROS_DEBUG("Adding feature points %lu", image.size());
//    if (solver_flag == INITIAL) {
        if (f_manager.addFeatureCheckParallax(frame_count, image, td))
            marginalization_flag = MARGIN_OLD;
        else
            marginalization_flag = MARGIN_SECOND_NEW;

//    } else {
//        if (f_manager.addFeatureCheckParallax(frame_count, image, td)
//            && (abs(AngleAxisd(tmp_pre_integration->delta_q).angle()) / (10 * M_PI / 180)
//                + (Ps[WINDOW_SIZE] - Ps[WINDOW_SIZE-2]).norm() / 0.2) >= 0.5)
//            marginalization_flag = MARGIN_OLD;
//        else
//            marginalization_flag = MARGIN_SECOND_NEW;
//
//    }

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    status.emplace_back("timestamp", Headers[frame_count].stamp.toSec());
    status.emplace_back("frame_count", frame_count);
    status.emplace_back("keyframe", marginalization_flag == MARGIN_OLD ? 0 : 1);
    status.emplace_back("feature_count", f_manager.getFeatureCount());
    if (tmp_pre_integration) {
        status_log_ypr("tmp_pre_integration", Utility::R2ypr(tmp_pre_integration->delta_q.toRotationMatrix()));
    }
    if (tmp_base_integration) {
        status_log_p("tmp_base_integration", tmp_base_integration->delta_p);
        status_log_ypr("tmp_base_integration", Utility::R2ypr(tmp_base_integration->delta_q.toRotationMatrix()));
    }

    /* stuff about all_image_frame */
    all_image_frame[header.stamp.toSec()] = ImageFrame(image, header.stamp.toSec());
    all_image_frame[header.stamp.toSec()].pre_integration = std::move(tmp_pre_integration);
    all_image_frame[header.stamp.toSec()].base_integration = std::move(tmp_base_integration);
    tmp_pre_integration.reset(new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]});
    tmp_base_integration.reset(new BaseOdometryIntegration3D(Bgs[frame_count]));

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // initialize VIO
        if (frame_count < WINDOW_SIZE) {
            ROS_DEBUG("Waiting for enough image frames to initialization (%d/%d)", frame_count, WINDOW_SIZE);
        } else {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1) {
                // try initialize
                result = initialStructure();
                initial_timestamp = header.stamp.toSec();
            }
            if(!result) {
                // initialize failed,
            } else {
                // initialize ok, do non-linear optimization on this frame
                base_integration_before_init.repropagate(rio * Vector3d(0, 0, (rio.transpose() * Bgs[0]).z()));
                wheel_imu_odom.repropagate(rio * Vector3d(0, 0, (rio.transpose() * Bgs[0]).z()));
                wheel_imu_odom.linearized_bg = {0, 0, 0};
                wheel_imu_odom3D.repropagate(Bgs[0]);
                wheel_imu_odom3D.linearized_bg = {0, 0, 0};

                solver_flag = NON_LINEAR;
                ROS_INFO("Initialization finish! skipped %.3f sec", base_integration_before_init.sum_dt);
            }
        }
    }

    if (solver_flag == NON_LINEAR) {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R  = Rs[WINDOW_SIZE];
        last_P  = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }

    status.emplace_back("solver_flag", solver_flag == NON_LINEAR ? 1 : 0);

    Affine3d T_origin_o;
    Vector3d vel_o;
    if (solver_flag == INITIAL) {
        T_origin_o = wheel_imu_odom.transform();
        vel_o.head<2>() = wheel_imu_odom.measurements.back().velocity.first;
        vel_o[2] = 0;
    } else {
        Affine3d T_b_o = Translation3d(tio) * rio;
        Vector3d ypr_o_b = Utility::R2ypr(rio.inverse());
        Affine3d T_worldbase_w = Translation3d(T_b_o.inverse().translation()) * Utility::ypr2R(Vector3d(ypr_o_b[0], 0, 0));
        Affine3d T_worldorigin_worldbase = Translation3d(base_integration_before_init.delta_p) * base_integration_before_init.delta_q;
        Affine3d T_w_b = Translation3d(Ps[frame_count]) * Rs[frame_count];
        T_origin_o = T_worldorigin_worldbase * T_worldbase_w * T_w_b * T_b_o;
        vel_o = (T_w_b * T_b_o).rotation().inverse() * Vs[frame_count];
    }
    status_log_p("base", T_origin_o);
    status_log_ypr("base", T_origin_o);
    status_log_p("base_vel", vel_o);

    total_distance += (T_origin_o.translation() - prev_position).norm();
    prev_position = T_origin_o.translation();
    status.emplace_back("total_distance", total_distance);

    status_log_p("bg", Bgs[frame_count]);
    status_log_p("ba", Bas[frame_count]);
    status.emplace_back("td", td);

    status_log_p("wheel_only_odom", wheel_only_odom.delta_p);
    status_log_ypr("wheel_only_odom", wheel_only_odom.delta_q);
    status_log_p("wheel_imu_odom", wheel_imu_odom.delta_p);
    status_log_ypr("wheel_imu_odom", wheel_imu_odom.delta_q);
    status_log_p("wheel_imu_odom3D", wheel_imu_odom3D.delta_p);
    status_log_ypr("wheel_imu_odom3D", wheel_imu_odom3D.delta_q);

    if (frame_count < WINDOW_SIZE) {
        frame_count++;
    } else {
        TicToc t_margin;
        slideWindow();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
    }
}

void Estimator::log_status() {
    std::map<string, double> data;
    for (int i = 0; i < status.size(); i++) {
        const string& key = status[i].first;
        double value = status[i].second;

        if (history_status.find(key) == history_status.end()) {
            if (i == 0)
                history_index.insert(history_index.begin(), key);
            else
                history_index.insert(std::next(std::find(history_index.begin(), history_index.end(), status[i-1].first)), key);

            if (history_status.empty()) {
                history_status[key] = {};
            } else {
                history_status[key] = std::vector<double>(
                        history_status.begin()->second.size(),
                        std::numeric_limits<double>::quiet_NaN());
            }
        }

        data[key] = value;
    }
    for (auto& kv : history_status) {
        try {
            kv.second.push_back(data.at(kv.first));
        } catch (const std::out_of_range&) {
            kv.second.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }
}
void Estimator::save_history(string path) {
    if (history_index.empty()) {
        ROS_INFO("No history need save");
        return;
    }
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        ROS_WARN_STREAM("Cannot open " << path << " to write history");
        return;
    }
    std::vector<const double*> data;
    std::ostringstream header;
    for (int i = 0; i < history_index.size(); i++) {
        const string& key = history_index[i];
        if (i < history_index.size() - 1) {
            ofs << key << ',';
            header << key << ',';
        } else {
            ofs << key << std::endl;
            header << key;
        }
        data.push_back(history_status[key].data());
    }
    ofs << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    for (int row = 0; row < history_status.begin()->second.size(); row++) {
        for (int col = 0; col < history_index.size(); col++) {
            if (col < history_index.size() - 1) {
                ofs << *data[col] << ',';
            } else {
                ofs << *data[col] << std::endl;
            }
            data[col]++;
        }
    }
    ofs.close();

    ROS_INFO("History saved to %s (%d cols, %d rows)",
             path.c_str(), (int)history_index.size(), (int)history_status.begin()->second.size());
}

// VIO initialize
bool Estimator::initialStructure()
{
    TicToc t_sfm;

    if (INIT_USE_ODOM == 1) {
        ROS_INFO("Initialize estimator using wheelodom+imu");

        // find previous frame which contians enough correspondance and parallex with newest frame
        bool enough_parallax = false;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            vector<pair<Vector3d, Vector3d>> corres;
            corres = f_manager.getCorresponding(i, WINDOW_SIZE);
            if (corres.size() > 20)
            {
                double sum_parallax = 0;
                double average_parallax;
                for (int j = 0; j < int(corres.size()); j++) {
                    Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                    Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                    double parallax = (pts_0 - pts_1).norm();
                    sum_parallax = sum_parallax + parallax;

                }
                average_parallax = 1.0 * sum_parallax / int(corres.size());
                if(average_parallax * 460 > 30) {
                    enough_parallax = true;
                }
            }
        }
        if (!enough_parallax) {
            ROS_INFO("Not enough features or parallax; Move device around");
            return false;
        }

        if (wheelOdomInitialAlign())
            return true;
        else
        {
            ROS_INFO("misalign wheel odom with IMU");
            return false;
        }
    }
    ROS_INFO("Initialize estimator using camera+imu");

    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;

        // get average acceleration (without gravity) in all frames
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);

        // get standard deviation of acceleration in every frame
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));

        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }

    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::wheelOdomInitialAlign() {
    TicToc t_g;

//    // check wheel odometry observibility
//    bool enough_move_ditance = true;
//    {
//        bool first = true;
//        double sum_distance = 0;
//        for (const auto& it : all_image_frame) {
//            if (first) {
//                first = false;
//                continue;
//            }
//            sum_distance += it.second.base_integration->delta_p.norm();
//        }
//        if (sum_distance < 0.5) {
//            ROS_INFO("move distance not enough (%f < 0.5)", sum_distance);
//            enough_move_ditance = false;
//        }
//    }
//
//    // check wheel odometry still frames
//    bool enough_still_duration = true;
//    {
//        double sum_still_dt = 0;
//        for (double k : GetStillFrames(all_image_frame, 1)) {
//            sum_still_dt += all_image_frame[k].base_integration->sum_dt;
//        }
//        if (sum_still_dt < 1) {
//            ROS_INFO("still duration not enough (%f < 1)", sum_still_dt);
//            enough_still_duration = false;
//        }
//    }

//    if (!enough_move_ditance) {
//        return false;
//    }

    VectorXd x;
    //solve scale
    bool result = WheelOdomIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_WARN("solve g failed!");
        return false;
    }
    ROS_INFO_STREAM("g0:         " << g.transpose());

    // init IMU pose
    Ps[0] = {0, 0, 0};
    Vector3d ypr_w_B0 = Utility::R2ypr(Utility::g2R(g));
    ypr_w_B0[0] = 0;
    Rs[0] = Utility::ypr2R(ypr_w_B0);
    ROS_INFO_STREAM("YPR^W_O0:   " << Utility::R2ypr(Rs[0] * rio).transpose());
    ROS_INFO_STREAM("Initial bg: " << Bgs[WINDOW_SIZE].transpose() * 180 / M_PI);

    g = Rs[0] * g;
    Affine3d T_B_O = Translation3d(tio) * rio;

    for (int i = 0; i <= WINDOW_SIZE; i++) {
        // repropagate with refined bias gyr
        if (pre_integrations[i])
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        if (base_integrations[i])
            base_integrations[i]->repropagate(Bgs[i]);
        // p & R comes from wheel odom and IMU
        if (i >= 1) {
            Affine3d T_w_Bi = Translation3d(Ps[i-1]) * Rs[i-1];
            Affine3d T_Oi_Oj = Translation3d(base_integrations[i]->delta_p) * base_integrations[i]->delta_q.toRotationMatrix();
            Affine3d T_w_Bj = T_w_Bi * T_B_O * T_Oi_Oj * T_B_O.inverse();
            Ps[i] = T_w_Bj.translation();
            Rs[i] = T_w_Bj.linear();
            all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
        }
        // set Vs to solved result
        Vs[i] = Rs[i] * x.segment<3>(i * 3);
    }

    // clear depth estimation of all features
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulate on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(tic[0]), &(ric[0]));

    return true;
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // T & R comes from global sfm (vision only)
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = all_image_frame[Headers[i].stamp.toSec()].T;
        Rs[i] = all_image_frame[Headers[i].stamp.toSec()].R;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // clear depth estimation of all features
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulate on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    // repropagate with refined bias gyr
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (pre_integrations[i])
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        if (base_integrations[i])
            base_integrations[i]->repropagate(Bgs[i]);
    }

    // apply scale and T_ic to P
    double s = (x.tail<1>())(0);
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);

    // set Vs to solved result
    int i = 0;
    for (auto& frame_i : all_image_frame) {
        if(frame_i.second.is_key_frame) {
            Vs[i] = frame_i.second.R * x.segment<3>(i * 3);
            i++;
        }
    }

    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // rotate everything to gravity frame (keep yaw direction of imu)
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

template <typename T>
static bool is_inside_periods(pair<double, double> t, const T& periods) {
    for (const auto& period : periods) {
        if (t.first < period.second && t.second > period.first) {
            return true;
        }
    }
    return false;
}

void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    /* check wheel slip */

    // linear alignment
    vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> frames;
    for (const auto& kv : all_image_frame) {
        if (kv.first <= Headers[0].stamp.toSec())
            continue;
        if (kv.first > Headers[frame_count-1].stamp.toSec())
            break;
        if (!kv.second.pre_integration || !kv.second.base_integration)
            continue;
        frames.emplace_back(make_pair(kv.second.pre_integration, kv.second.base_integration));
    }
    vector<Vector3d> align_predict_err_ps, align_err_vs;
    VectorXd x;
    Vector3d g = Rs[0].inverse() * Vector3d(0, 0, G.norm());
    base_imu_alignment_fixed_scale_g(frames, rio, tio, x, g, align_predict_err_ps, align_err_vs);

    double sum_dt = 0;
    Vector3d acc_0, gyr_0;
    Vector3d P; P.setZero();
    Quaterniond Q; Q.setIdentity();
    Vector3d V = x.tail<3>();
    for (const auto& kv : all_image_frame) {
        if (kv.first <= Headers[frame_count-1].stamp.toSec())
            continue;
        const auto& pre = *kv.second.pre_integration;
        acc_0 = pre.linearized_acc;
        gyr_0 = pre.linearized_gyr;
        for (int i = 0; i < pre.dt_buf.size(); i++) {
            const auto& acc_1 = pre.acc_buf[i];
            const auto& gyr_1 = pre.gyr_buf[i];
            ImuUtility::midpoint_integration(pre.dt_buf[i], Rs[frame_count-1].inverse() * Vector3d(0, 0, G.norm()),
                                             acc_0 - pre.linearized_ba, gyr_0 - pre.linearized_bg,
                                             acc_1 - pre.linearized_ba, gyr_1 - pre.linearized_bg,
                                             P, Q, V);
            sum_dt += pre.dt_buf[i];
            acc_0 = acc_1;
            gyr_0 = gyr_1;
        }
    }
    ROS_ASSERT(abs(sum_dt - (Headers[frame_count] - Headers[frame_count-1]).toSec()) < 0.001);
    Affine3d align_predict_T_Bi_Bj = Translation3d(P) * Q;

    Affine3d T_w_Bi = Translation3d(Ps[frame_count-1]) * Rs[frame_count-1];
    Affine3d T_w_Bj = Translation3d(Ps[frame_count]) * Rs[frame_count];
    Affine3d estimator_predict_T_Bi_Bj = T_w_Bi.inverse() * T_w_Bj;
    
    Affine3d T_B_O = Translation3d(tio) * rio;
    Affine3d align_predict_T_Oi_Oj = T_B_O.inverse() * align_predict_T_Bi_Bj * T_B_O;
    Affine3d estimator_predict_T_Oi_Oj = T_B_O.inverse() * estimator_predict_T_Bi_Bj * T_B_O;
    Affine3d wheel_predict_T_Oi_Oj = Translation3d(wheel_imu_predict->delta_p) * wheel_imu_predict->delta_q.toRotationMatrix();
    
    Vector3d align_predict_err_p = align_predict_err_ps.back();
    Vector3d estimator_predict_err_p = estimator_predict_T_Oi_Oj.translation() - wheel_predict_T_Oi_Oj.translation();
    Quaterniond estimator_predict_err_q(wheel_predict_T_Oi_Oj.linear().inverse() * estimator_predict_T_Oi_Oj.linear());

    Vector3d predict_std(sqrt(wheel_imu_predict->covariance(0, 0)),
                 sqrt(wheel_imu_predict->covariance(1, 1)),
                 sqrt(wheel_imu_predict->covariance(2, 2)));
    double ma_dist_predict = sqrt(estimator_predict_err_p.transpose() * wheel_imu_predict->covariance.block<3, 3>(0, 0).inverse() * estimator_predict_err_p);
    double ma_dist_align = sqrt(align_predict_err_p.transpose() * wheel_imu_predict->covariance.block<3, 3>(0, 0).inverse() * align_predict_err_p);

    char c_str[1024];
    snprintf(c_str, 1024,
             "%5.3f-%5.3f=%5.3lf \n"
             "predict_err[%6.3f %6.3f %6.3f] predict_ma_dist=%5.3lf \n"
             "  align_err[%6.3f %6.3f %6.3f]   align_ma_dist=%5.3lf \n"
             "predict_std[%6.3f %6.3f %6.3f]     min_ma_dist=%5.3lf ",
             estimator_predict_T_Oi_Oj.translation().norm(), wheel_predict_T_Oi_Oj.translation().norm(), estimator_predict_err_p.norm(),
             estimator_predict_err_p.x(), estimator_predict_err_p.y(), estimator_predict_err_p.z(), ma_dist_predict,
             align_predict_err_p.x(), align_predict_err_p.y(), align_predict_err_p.z(), ma_dist_align,
             predict_std.x(), predict_std.y(), predict_std.z(),
             min(ma_dist_predict, ma_dist_align));
    string info(c_str);

    if (!slipping && min(ma_dist_predict, ma_dist_align) > 1.5 && (Headers[WINDOW_SIZE].stamp.toSec() - initial_timestamp) > 1) {
        slipping = true;
    } else if (slipping && max(ma_dist_predict, ma_dist_align) <= 1) {
        if (USE_ODOM)
            ROS_INFO_STREAM("Wheel normal! " << info);
        else
            ROS_DEBUG_STREAM("Wheel normal! " << info);
        slipping = false;
    }
    if (slipping) {
        wheel_slip_periods.emplace_back(
                Headers[WINDOW_SIZE-1].stamp.toSec(),
                Headers[WINDOW_SIZE].stamp.toSec());
        if (USE_ODOM)
            ROS_WARN_STREAM("Wheel slip! " << info);
        else
            ROS_DEBUG_STREAM("Wheel slip! " << info);
        status.emplace_back("slip", 1);
    } else {
        status.emplace_back("slip", 0);
    }

    status_log_p("align_err", align_predict_err_p);
    status.emplace_back("align_ma_dist", ma_dist_align);
    status_log_p("predict_err", estimator_predict_err_p);
    status.emplace_back("predict_ma_dist", ma_dist_predict);
    status_log_p("predict_std", predict_std);

    wheel_imu_predict.reset();

    if (USE_ODOM) {
        if (USE_ODOM == 1 || USE_ODOM == 3) {
            ostringstream oss;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                int j = i + 1;
                if (!is_inside_periods({Headers[i].stamp.toSec(), Headers[j].stamp.toSec()}, wheel_slip_periods))
                {
                    BaseOdomFactor *base_odom_factor = new BaseOdomFactor(base_integrations[j]);
                    problem.AddResidualBlock(base_odom_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j]);
                } else {
                    oss << i << "-" << j << " ";
                }
            }
            if (!oss.str().empty())
                ROS_DEBUG("Igonre wheelodom %s", oss.str().c_str());
        }
        if (USE_ODOM == 2 || USE_ODOM == 3) {
            ostringstream oss;
            auto long_base_integration = std::make_shared<BaseOdometryIntegration3D>(Bgs[0]);
            int begin = 0;
            int end = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                int j = i + 1;
                if (!is_inside_periods({Headers[i].stamp.toSec(), Headers[j].stamp.toSec()}, wheel_slip_periods))
                {
                    // this period not slip
                    for (const auto &m : base_integrations[j]->measurements) {
                        long_base_integration->push_back(m);
                    }
                    end = j;
                } else {
                    // this period slip
                    if (end - begin > 1) {
                        BaseOdomFactor *base_odom_factor = new BaseOdomFactor(long_base_integration);
                        problem.AddResidualBlock(base_odom_factor, NULL, para_Pose[begin], para_SpeedBias[begin],
                                                 para_Pose[end]);
                        oss << begin << "-" << end << " ";
                    }
                    long_base_integration = std::make_shared<BaseOdometryIntegration3D>(Bgs[0]);
                    begin = j;
                }
            }
            if (end - begin > 1) {
                BaseOdomFactor *base_odom_factor = new BaseOdomFactor(long_base_integration);
                problem.AddResidualBlock(base_odom_factor, NULL, para_Pose[begin], para_SpeedBias[begin],
                                         para_Pose[end]);
                oss << begin << "-" << end << " ";
            }
            if (begin > 0 || end < WINDOW_SIZE) {
                ROS_DEBUG_STREAM("Long factor: " << oss.str());
            }
        }
    }

    if (USE_PLANE_FACTOR) {
        for (int i = 0; i <= WINDOW_SIZE; i++)
            problem.AddResidualBlock(new GlobalPlaneFactor(), NULL, para_Pose[i]);
    }

    if (true || f_manager.last_track_num >= 100) {
        int f_m_cnt = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                {
                    continue;
                }
                Vector3d pts_j = it_per_frame.point;
                if (ESTIMATE_TD)
                {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                      it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                      it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
                }
                else
                {
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                }
                f_m_cnt++;
            }
        }
        ROS_DEBUG("visual measurement count: %d ", f_m_cnt);
//        ROS_INFO("visual (%d points)", f_manager.last_track_num);
    } else {
        ROS_WARN("visual lost (%d points), visual data ignored ", f_manager.last_track_num);
    }

    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }

    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 4;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations: %d, solver costs: %f", static_cast<int>(summary.iterations.size()), t_solver.toc());

    double2vector();

    // evaluate odom residual
//    if (USE_ODOM == 1 || USE_ODOM == 3)
//        for (int i = 0; i < WINDOW_SIZE; i++)
//        {
//            int j = i + 1;
//    //        ROS_INFO_STREAM("P covariance:" << std::endl << base_integrations[j]->covariance);
//            base_integrations[j]->evaluate(Ps[i], Quaterniond(Rs[i]), Bgs[i], Ps[j], Quaterniond(Rs[j]), true);
//        }
//    if (USE_ODOM == 2 || USE_ODOM == 3) {
//        long_base_integration->evaluate(Ps[0], Quaterniond(Rs[0]), Bgs[0], Ps[WINDOW_SIZE], Quaterniond(Rs[WINDOW_SIZE]), true);
//    }

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info =
                        new ResidualBlockInfo(imu_factor, NULL,
                                              vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                              vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            if (USE_ODOM && !is_inside_periods({Headers[0].stamp.toSec(), Headers[1].stamp.toSec()}, wheel_slip_periods)) {
                BaseOdomFactor *base_odom_factor = new BaseOdomFactor(base_integrations[1]);
                ResidualBlockInfo *residual_block_info =
                        new ResidualBlockInfo(base_odom_factor, NULL,
                                              vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1]},
                                              vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    // debug
//    wheel_imu_P = Ps[WINDOW_SIZE];
//    wheel_imu_V = Rs[WINDOW_SIZE] * rio * Eigen::Vector3d(base_integrations[WINDOW_SIZE]->measurements.back().velocity.first.x(),
//                                        base_integrations[WINDOW_SIZE]->measurements.back().velocity.first.y(), 0);

    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        // marginalize oldest frame
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);
                std::swap(base_integrations[i], base_integrations[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]);
            base_integrations[WINDOW_SIZE] = std::make_shared<BaseOdometryIntegration3D>(Bgs[WINDOW_SIZE]);

            double t_0 = Headers[0].stamp.toSec();
            auto it_0 = all_image_frame.find(t_0);

            if (solver_flag == INITIAL) {
                int n = 0;
                for (auto it = all_image_frame.begin(); it != it_0; it++) {
                    if (it->second.base_integration) {
                        for (const auto &m : it->second.base_integration->measurements) {
                            base_integration_before_init.push_back(MixedOdomMeasurement(m.dt, m.velocity, m.constraint_error_vel,
                                    rio * Vector3d(0, 0, (rio.inverse() * m.imu_angular_velocity ).z()),
                                    m.imu_linear_acceleration));
                        }
                        n++;
                    }
                }
                ROS_DEBUG("Throw image frame before init: %d", n);
            }
            // remove image frames older than first keyframe
            all_image_frame.erase(all_image_frame.begin(), it_0);

            while (!wheel_slip_periods.empty() && wheel_slip_periods.front().second < t_0)
                wheel_slip_periods.erase(wheel_slip_periods.begin());

            slideWindowOld();
        }
    }
    else
    {
        // marginalize second new frame
        if (frame_count == WINDOW_SIZE)
        {
            // only do marginalization if WINDOW is full

            // move imu measurement from newest frame to second newest frame
            for (unsigned int i = 0; i < pre_integrations[frame_count]->dt_buf.size(); i++)
            {
                double tmp_dt = pre_integrations[frame_count]->dt_buf[i];
                Vector3d tmp_linear_acceleration = pre_integrations[frame_count]->acc_buf[i];
                Vector3d tmp_angular_velocity = pre_integrations[frame_count]->gyr_buf[i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
            }
            for (auto& m : base_integrations[frame_count]->measurements) {
                base_integrations[frame_count-1]->push_back(m);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]);
            base_integrations[WINDOW_SIZE] = std::make_shared<BaseOdometryIntegration3D>(Bgs[WINDOW_SIZE]);

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

