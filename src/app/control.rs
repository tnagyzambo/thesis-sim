use anyhow::Result;
use na::{Matrix3, Quaternion, UnitQuaternion, Vector3, Vector4};
use nalgebra as na;

use super::{plot, q_ln, Force, State, Torque, A, J, M, MOTOR_A, MOTOR_B, Q_INVERT};

const K_Q: f64 = 50.0;
const K_W: f64 = 2.0;
const K_P: f64 = 1.0;
const K_D: f64 = 2.0;
const K_1: f64 = 0.33;
const K_2: f64 = 1.0;

#[derive(Default)]
pub struct ControllerState {
    wdot_d: Vector3<f64>,
}

pub fn control(
    rec: &rerun::RecordingStream,
    state: &State,
    controller_state: &mut ControllerState,
    motor_state: &mut Vector4<f64>,
    dt: f64,
    t: f64,
) -> Result<(Force, Torque)> {
    // POS TARGETS
    let p_t = Vector3::<f64>::new(0.0, 0.0, 0.0);
    let pdot_t = Vector3::<f64>::zeros();

    // GUIDANCE
    let e_n = Q_INVERT * (p_t - state.position());
    let edot_n = Q_INVERT * (pdot_t - state.velocity());
    let e_d = Vector3::<f64>::new(0.0, 0.0, e_n.norm());
    let q_d = if e_d.cross(&e_n).norm() <= 0.0 {
        UnitQuaternion::<f64>::identity()
    } else {
        let theta = K_1 * (K_2 * e_n.norm()).atan();
        let axis = e_n.cross(&e_d).normalize();
        UnitQuaternion::new_normalize(Quaternion::from_parts(
            (theta / 2.0).cos(),
            axis * (theta / 2.0).sin(),
        ))
    };
    let skew = if e_n.norm() <= 0.001 {
        Matrix3::<f64>::zeros()
    } else {
        Matrix3::<f64>::new(
            0.0,
            -1.0 / e_n.norm(),
            0.0, //
            1.0 / e_n.norm(),
            0.0,
            0.0, //
            0.0,
            0.0,
            0.0, //
        )
    };

    let w_d = skew * -(q_d.conjugate() * edot_n);
    let wdot_d = (w_d - controller_state.wdot_d) / dt;
    controller_state.wdot_d = w_d;

    // ATTITUDE CONTROLLER
    let q_e = q_ln(state.attitude() * q_d.conjugate());
    let w_e = state.rate() - ((state.attitude() * q_d.conjugate()).conjugate() * w_d);
    let wdot_e = state.attitude() * (q_d.conjugate() * wdot_d);

    let tau_u = state.rate().cross(&(J * state.rate()))
                    + J * wdot_e
                    //- K_Q * q_e.imag()
                    - K_Q * q_e
                    - K_W * w_e;

    // TRANSLATIONAL CONTROLLER
    let p_e = p_t - state.position();
    let pdot_e = pdot_t - state.velocity();

    let f_thrust = (Vector3::new(0.0, 0.0, M * 9.81) + K_P * p_e + K_D * pdot_e)[2];

    let (f_thrust, tau_u) = motor_tf(f_thrust, tau_u, motor_state, dt, &rec, t)?;

    let f_u = Force::new(Vector3::new(0.0, 0.0, -f_thrust), "u".to_string());
    let tau_u = Torque::new(tau_u, "u".to_string());

    Ok((f_u, tau_u))
}

fn motor_tf(
    f: f64,
    tau: Vector3<f64>,
    motor_state: &mut Vector4<f64>,
    dt: f64,
    rec: &rerun::RecordingStream,
    t: f64,
) -> Result<(f64, Vector3<f64>)> {
    let a_inv = A.try_inverse().unwrap();
    let force_vec = Vector4::new(f, tau[0], tau[1], tau[2]);
    let rpm_t_squared = a_inv * force_vec;
    let rpm_t = Vector4::new(
        rpm_t_squared[0].max(0.0).sqrt(),
        rpm_t_squared[1].max(0.0).sqrt(),
        rpm_t_squared[2].max(0.0).sqrt(),
        rpm_t_squared[3].max(0.0).sqrt(),
    );

    let motor_state_dot = MOTOR_A * *motor_state + MOTOR_B * rpm_t;
    *motor_state = *motor_state + (dt * motor_state_dot);

    let rpm = motor_state;

    plot::plot_rotor(rec, &rpm_t, rpm, t)?;

    let rpm_squared = Vector4::new(
        rpm[0].powi(2),
        rpm[1].powi(2),
        rpm[2].powi(2),
        rpm[3].powi(2),
    );

    let forces = A * rpm_squared;

    let f = forces[0];
    let tau = Vector3::new(forces[1], forces[2], forces[3]);

    Ok((f, tau))
}
