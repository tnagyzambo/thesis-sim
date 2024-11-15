use super::{dq_exp, Force, State, Torque, M};
use na::Vector3;
use nalgebra as na;

pub fn dynamics(state: &State, forces: &Vec<Force>, torques: &Vec<Torque>, dt: f64) -> State {
    let tau: Vec<Vector3<f64>> = torques.iter().map(|torque| torque.tau).collect();
    let f: Vec<Vector3<f64>> = forces.iter().map(|force| force.f).collect();

    // eta is not constrained by the unit norm
    let xi = state.xi + (dt * state.compute_wrench(tau.as_slice(), f.as_slice()));

    // Exponential intergration to maintain unit norm of q
    let q = state.q * dq_exp(0.5 * dt * state.xi);

    State { q, xi }
}

pub fn ficticous_forces(state: &State) -> Vec<Force> {
    let f_coriolis = Force::new(
        -2.0 * M * state.rate().cross(&state.velocity_body()),
        "coriolis".to_string(),
    );

    let f_centrifugal = Force::new(
        -M * state
            .rate()
            .cross(&state.rate().cross(&state.position_body())),
        "centrifugal".to_owned(),
    );

    //let f_euler = Force::new(
    //    -M * wdot_body.cross(&state.position_body()),
    //    "euler".to_string(),
    //);

    vec![
        f_centrifugal,
        f_coriolis,
        //f_euler,
    ]
}

pub fn gravity(state: &State) -> Force {
    Force::new(
        M * (state.attitude().conjugate() * Vector3::<f64>::new(0.0, 0.0, 9.81)),
        "g".to_string(),
    )
}
