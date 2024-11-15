use super::state::{State, Q_INVERT};
use super::{Force, Torque};
use anyhow::Result;
use na::{Quaternion, UnitQuaternion, Vector3, Vector4};
use nalgebra as na;

const COLOR_X: [u8; 3] = [254, 76, 66];
const COLOR_Y: [u8; 3] = [87, 182, 70];
const COLOR_Z: [u8; 3] = [40, 208, 209];
const COLOR_0: [u8; 3] = [211, 166, 35];

pub fn plot_all_static(rec: &rerun::RecordingStream) -> Result<()> {
    rec.set_time_seconds("sim_time", 0);

    // 3D statics
    drone_static(&rec, "3d/world/drone")?;
    triade_static(&rec, "3d/world/inertial_frame", 1.0, 255)?;
    triade_static(&rec, "3d/world/body_frame", 0.1, 255)?;
    triade_static(&rec, "3d/world/target_frame", 0.1, 100)?;

    // Timeseries styling
    styling_quaternion_parameters(&rec, "q/q", 255, 1.0)?;
    styling_quaternion_parameters(&rec, "q/t", 100, 0.8)?;
    styling_rotational(&rec, "attitude/q", 255, 1.0)?;
    styling_rotational(&rec, "attitude/t", 100, 0.8)?;
    styling_rotational(&rec, "rate/w", 255, 1.0)?;
    styling_rotational(&rec, "rate/t", 100, 0.8)?;
    styling_rotational(&rec, "torque/body/u", 100, 0.8)?;
    styling_cartesian(&rec, "position/world/r", 255, 1.0)?;
    styling_cartesian(&rec, "position/world/t", 100, 0.8)?;
    styling_cartesian(&rec, "velocity/world/v", 255, 1.0)?;
    styling_cartesian(&rec, "velocity/world/t", 100, 0.8)?;
    styling_cartesian(&rec, "position/body/r", 255, 1.0)?;
    styling_cartesian(&rec, "position/body/t", 100, 0.8)?;
    styling_cartesian(&rec, "velocity/body/v", 255, 1.0)?;
    styling_cartesian(&rec, "velocity/body/t", 100, 0.8)?;
    styling_cartesian(&rec, "force/body/centrifugal", 100, 0.8)?;
    styling_cartesian(&rec, "force/body/coriolis", 100, 0.8)?;
    styling_cartesian(&rec, "force/body/euler", 100, 0.8)?;
    styling_cartesian(&rec, "force/body/g", 100, 0.8)?;
    styling_cartesian(&rec, "force/body/u", 100, 0.8)?;

    Ok(())
}

pub fn plot_all(
    rec: &rerun::RecordingStream,
    state: &State,
    target_attitude: &UnitQuaternion<f64>,
    target_rate: &Vector3<f64>,
    target_position: &Vector3<f64>,
    error_position: &Vector3<f64>,
    error_velocity: &Vector3<f64>,
    wdot_body: &Vector3<f64>,
    tau: &[(Vector3<f64>, &str)],
    f: &[(Vector3<f64>, &str)],
    t: f64,
) -> Result<()> {
    // Set timestep
    rec.set_time_seconds("sim_time", t);

    // Plot drone
    update_static_with_pose(rec, "3d/world/drone", &state.rotation(), &state.position())?;
    update_static_with_pose(
        rec,
        "3d/world/body_frame",
        &state.rotation(),
        &state.position(),
    )?;
    update_static_with_pose(
        rec,
        "3d/world/target_frame",
        &(Q_INVERT * target_attitude),
        &target_position,
    )?;

    // Plot rotational components
    quaternion_parameters(rec, "q/q", &state.attitude())?;
    quaternion_parameters(rec, "q/t", &target_attitude)?;
    {
        let (roll, pitch, yaw) = state.attitude().euler_angles();
        let a = Vector3::new(roll, pitch, yaw);
        rotational(rec, "attitude/q", &a)?;
    }
    {
        let (roll, pitch, yaw) = target_attitude.euler_angles();
        let a_t = Vector3::new(roll, pitch, yaw);
        rotational(rec, "attitude/t", &a_t)?;
    }
    rotational(rec, "rate/w", &state.rate())?;
    rotational(rec, "rate/t", target_rate)?;
    rotational(rec, "angular_accl/wdot", wdot_body)?;

    // Plot translational components
    cartesian(rec, "position/world/r", &state.position())?;
    //cartesian(rec, "position/world/t", &r_t)?;
    cartesian(rec, "velocity/world/v", &state.velocity())?;
    //cartesian(rec, "velocity/world/t", &v_t)?;
    cartesian(rec, "position/body/r", &state.position_body())?;
    //cartesian(rec, "position/body/t", &r_t)?;
    cartesian(rec, "velocity/body/v", &state.velocity_body())?;
    //cartesian(rec, "velocity/body/t", &v_t)?;

    // Plot forces
    for (torque, tag) in tau.iter() {
        rotational(rec, format!("torque/body/{}", tag).as_str(), torque)?;
    }
    for (force, tag) in f.iter() {
        cartesian(rec, format!("force/body/{}", tag).as_str(), force)?;
    }

    // Plot position in body
    rec.log(
        "3d/world/e",
        &rerun::Arrows3D::from_vectors(&[
            (Q_INVERT.conjugate() * error_position).into_rerun_vec3d()
        ])
        .with_origins(&[state.position().into_rerun_vec3d()]),
    )?;
    rec.log(
        "3d/world/edot",
        &rerun::Arrows3D::from_vectors(&[
            (Q_INVERT.conjugate() * error_velocity).into_rerun_vec3d()
        ])
        .with_origins(&[state.position().into_rerun_vec3d()]),
    )?;

    Ok(())
}

pub fn plot_state(rec: &rerun::RecordingStream, state: &State, t: f64) -> Result<()> {
    // Set timestep
    rec.set_time_seconds("sim_time", t);

    // Plot drone
    update_static_with_pose(rec, "3d/world/drone", &state.rotation(), &state.position())?;
    update_static_with_pose(
        rec,
        "3d/world/body_frame",
        &state.rotation(),
        &state.position(),
    )?;

    // Plot rotational components
    quaternion_parameters(rec, "q/q", &state.attitude())?;
    {
        let (roll, pitch, yaw) = state.attitude().euler_angles();
        let a = Vector3::new(roll, pitch, yaw);
        rotational(rec, "attitude/q", &a)?;
    }
    rotational(rec, "rate/w", &state.rate())?;

    // Plot translational components
    cartesian(rec, "position/world/r", &state.position())?;
    cartesian(rec, "velocity/world/v", &state.velocity())?;
    cartesian(rec, "position/body/r", &state.position_body())?;
    cartesian(rec, "velocity/body/v", &state.velocity_body())?;

    Ok(())
}

pub fn plot_noisy_state(rec: &rerun::RecordingStream, state: &State, t: f64) -> Result<()> {
    // Set timestep
    rec.set_time_seconds("sim_time", t);

    // Plot rotational components
    quaternion_parameters(rec, "q/q_n", &state.attitude())?;
    {
        let (roll, pitch, yaw) = state.attitude().euler_angles();
        let a = Vector3::new(roll, pitch, yaw);
        rotational(rec, "attitude/q_n", &a)?;
    }
    rotational(rec, "rate/w_n", &state.rate())?;

    // Plot translational components
    cartesian(rec, "position/world/r_n", &state.position())?;
    cartesian(rec, "velocity/world/v_n", &state.velocity())?;
    cartesian(rec, "position/body/r_n", &state.position_body())?;
    cartesian(rec, "velocity/body/v_n", &state.velocity_body())?;

    Ok(())
}

pub fn plot_forces(
    rec: &rerun::RecordingStream,
    forces: &Vec<Force>,
    torques: &Vec<Torque>,
    t: f64,
) -> Result<()> {
    // Set timestep
    rec.set_time_seconds("sim_time", t);

    // Plot forces
    for torque in torques.iter() {
        rotational(
            rec,
            format!("torque/body/{}", torque.name).as_str(),
            &torque.tau,
        )?;
    }
    for force in forces.iter() {
        cartesian(rec, format!("force/body/{}", force.name).as_str(), &force.f)?;
    }

    Ok(())
}

pub fn plot_rotor(
    rec: &rerun::RecordingStream,
    rpm_t: &Vector4<f64>,
    rpm: &Vector4<f64>,
    t: f64,
) -> Result<()> {
    // Set timestep
    rec.set_time_seconds("sim_time", t);

    // Plot rotor speeds
    rec.log("rotor/1/t", &rerun::Scalar::new(rpm_t[0]))?;
    rec.log("rotor/2/t", &rerun::Scalar::new(rpm_t[1]))?;
    rec.log("rotor/3/t", &rerun::Scalar::new(rpm_t[2]))?;
    rec.log("rotor/4/t", &rerun::Scalar::new(rpm_t[3]))?;
    rec.log("rotor/1/w", &rerun::Scalar::new(rpm[0]))?;
    rec.log("rotor/2/w", &rerun::Scalar::new(rpm[1]))?;
    rec.log("rotor/3/w", &rerun::Scalar::new(rpm[2]))?;
    rec.log("rotor/4/w", &rerun::Scalar::new(rpm[3]))?;

    Ok(())
}

pub fn quaternion_parameters(
    rec: &rerun::RecordingStream,
    tag: &str,
    q: &UnitQuaternion<f64>,
) -> Result<()> {
    rec.log(format!("{}/1", tag), &rerun::Scalar::new(q.as_vector()[0]))?;
    rec.log(format!("{}/2", tag), &rerun::Scalar::new(q.as_vector()[1]))?;
    rec.log(format!("{}/3", tag), &rerun::Scalar::new(q.as_vector()[2]))?;
    rec.log(format!("{}/0", tag), &rerun::Scalar::new(q.as_vector()[3]))?;
    Ok(())
}

pub fn rotational(rec: &rerun::RecordingStream, tag: &str, w: &Vector3<f64>) -> Result<()> {
    rec.log(format!("{}/roll", tag), &rerun::Scalar::new(w[0]))?;
    rec.log(format!("{}/pitch", tag), &rerun::Scalar::new(w[1]))?;
    rec.log(format!("{}/yaw", tag), &rerun::Scalar::new(w[2]))?;
    Ok(())
}

pub fn cartesian(rec: &rerun::RecordingStream, tag: &str, r: &Vector3<f64>) -> Result<()> {
    rec.log(format!("{}/x", tag), &rerun::Scalar::new(r[0]))?;
    rec.log(format!("{}/y", tag), &rerun::Scalar::new(r[1]))?;
    rec.log(format!("{}/z", tag), &rerun::Scalar::new(r[2]))?;
    Ok(())
}

pub fn styling_quaternion_parameters(
    rec: &rerun::RecordingStream,
    tag: &str,
    alpha: u8,
    width: f32,
) -> Result<()> {
    rec.log_static(
        format!("{}/1", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_X[0], COLOR_X[1], COLOR_X[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/2", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Y[0], COLOR_Y[1], COLOR_Y[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/3", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Z[0], COLOR_Z[1], COLOR_Z[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/0", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_0[0], COLOR_0[1], COLOR_0[2], alpha])
            .with_width(width),
    )?;
    Ok(())
}

pub fn styling_rotational(
    rec: &rerun::RecordingStream,
    tag: &str,
    alpha: u8,
    width: f32,
) -> Result<()> {
    rec.log_static(
        format!("{}/roll", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_X[0], COLOR_X[1], COLOR_X[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/pitch", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Y[0], COLOR_Y[1], COLOR_Y[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/yaw", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Z[0], COLOR_Z[1], COLOR_Z[2], alpha])
            .with_width(width),
    )?;
    Ok(())
}

pub fn styling_cartesian(
    rec: &rerun::RecordingStream,
    tag: &str,
    alpha: u8,
    width: f32,
) -> Result<()> {
    rec.log_static(
        format!("{}/x", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_X[0], COLOR_X[1], COLOR_X[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/y", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Y[0], COLOR_Y[1], COLOR_Y[2], alpha])
            .with_width(width),
    )?;
    rec.log_static(
        format!("{}/z", tag),
        &rerun::SeriesLine::new()
            .with_color([COLOR_Z[0], COLOR_Z[1], COLOR_Z[2], alpha])
            .with_width(width),
    )?;
    Ok(())
}

pub fn drone_static(rec: &rerun::RecordingStream, tag: &str) -> Result<()> {
    rec.log_static(tag, &rerun::Asset3D::from_file("assets/drone.stl")?)?;
    Ok(())
}

pub fn triade_static(rec: &rerun::RecordingStream, tag: &str, scale: f64, alpha: u8) -> Result<()> {
    let x = Vector3::<f64>::x_axis().into_inner() * scale;
    let y = Vector3::<f64>::y_axis().into_inner() * scale;
    let z = Vector3::<f64>::z_axis().into_inner() * scale;

    rec.log_static(
        tag,
        &rerun::Arrows3D::from_vectors(&[
            x.into_rerun_vec3d(),
            y.into_rerun_vec3d(),
            z.into_rerun_vec3d(),
        ])
        .with_origins(&[[0.0, 0.0, 0.0]])
        .with_colors([[255, 0, 0, alpha], [0, 255, 0, alpha], [0, 0, 255, alpha]]),
    )?;
    Ok(())
}

pub fn plot_scalar(rec: &rerun::RecordingStream, scalar: f64, tag: &str, t: f64) -> Result<()> {
    rec.set_time_seconds("sim_time", t);
    rec.log(tag, &rerun::Scalar::new(scalar))?;
    Ok(())
}

pub fn plot_vec(rec: &rerun::RecordingStream, vec: Vector3<f64>, tag: &str, t: f64) -> Result<()> {
    rec.set_time_seconds("sim_time", t);
    rec.log(
        tag,
        &rerun::Arrows3D::from_vectors(&[vec.into_rerun_vec3d()]).with_origins(&[[0.0, 0.0, 0.0]]),
    )?;
    Ok(())
}

pub fn update_static_with_pose(
    rec: &rerun::RecordingStream,
    tag: &str,
    q: &UnitQuaternion<f64>,
    r: &Vector3<f64>,
) -> Result<()> {
    rec.log(
        tag,
        &rerun::InstancePoses3D::new()
            .with_quaternions([q.into_rerun_quaternion()])
            .with_translations([r.into_rerun_vec3d()]),
    )?;
    Ok(())
}

pub fn update_static_with_rotation(
    rec: &rerun::RecordingStream,
    tag: &str,
    q: UnitQuaternion<f64>,
) -> Result<()> {
    rec.log(tag, &q.into_rerun_instance_pose_3d())?;
    Ok(())
}

trait IntoRerunInstancePose3D {
    fn into_rerun_instance_pose_3d(&self) -> rerun::InstancePoses3D;
}

trait IntoRerunQuaternion {
    fn into_rerun_quaternion(&self) -> rerun::Quaternion;
}

impl IntoRerunQuaternion for Quaternion<f64> {
    fn into_rerun_quaternion(&self) -> rerun::Quaternion {
        rerun::Quaternion::from_xyzw(std::array::from_fn(|i| self[i] as f32))
    }
}

impl IntoRerunInstancePose3D for UnitQuaternion<f64> {
    fn into_rerun_instance_pose_3d(&self) -> rerun::InstancePoses3D {
        rerun::InstancePoses3D::new().with_quaternions([self.into_rerun_quaternion()])
    }
}

trait IntoRerunVec3D {
    fn into_rerun_vec3d(&self) -> rerun::Vec3D;
}

impl IntoRerunVec3D for Vector3<f64> {
    fn into_rerun_vec3d(&self) -> rerun::Vec3D {
        rerun::Vec3D::from(std::array::from_fn(|i| self[i] as f32))
    }
}
