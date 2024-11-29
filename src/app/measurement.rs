use super::State;
use na::Vector3;
use nalgebra as na;
use rand_distr::{Distribution, Normal};

pub fn measurment(
    state: &State,
    accl: Vector3<f64>,
    bias: &mut Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    // GPS Position
    let pos_normal = Normal::new(0.0, 0.5).unwrap();
    let pos_noise = Vector3::new(
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_pos = state.position() + pos_noise;

    // GPS Velocity
    let vel_normal = Normal::new(0.0, 0.5).unwrap();
    let vel_noise = Vector3::new(
        vel_normal.sample(&mut rand::thread_rng()),
        vel_normal.sample(&mut rand::thread_rng()),
        vel_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_vel = state.velocity() + vel_noise;

    // Acceleration
    let accl_normal = Normal::new(0.0, 0.1).unwrap();
    let accl_noise = Vector3::new(
        accl_normal.sample(&mut rand::thread_rng()),
        accl_normal.sample(&mut rand::thread_rng()),
        accl_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_accl = accl + accl_noise;

    // Rates
    let bias_normal = Normal::new(0.0, 0.01).unwrap();
    let bias_noise = Vector3::new(
        bias_normal.sample(&mut rand::thread_rng()),
        bias_normal.sample(&mut rand::thread_rng()),
        bias_normal.sample(&mut rand::thread_rng()),
    );
    *bias += bias_noise;

    let rate_normal = Normal::new(0.0, 0.01).unwrap();
    let rate_noise = Vector3::new(
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_rate = state.rate() + rate_noise + *bias;

    (noisy_pos, noisy_vel, noisy_accl, noisy_rate)
}
