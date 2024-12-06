use super::State;
use na::Vector3;
use nalgebra as na;
use rand_distr::{Distribution, Normal};

pub fn measurment(
    state: &State,
    accl: Vector3<f64>,
    rate_bias: &mut Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    // Acceleration
    let accl_normal = Normal::new(0.0, 0.000015).unwrap();
    let accl_noise = Vector3::new(
        accl_normal.sample(&mut rand::thread_rng()),
        accl_normal.sample(&mut rand::thread_rng()),
        accl_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_accl = accl + accl_noise;

    // Rates
    let rate_bias_normal = Normal::new(0.0, 0.01).unwrap();
    let rate_bias_noise = Vector3::new(
        rate_bias_normal.sample(&mut rand::thread_rng()),
        rate_bias_normal.sample(&mut rand::thread_rng()),
        rate_bias_normal.sample(&mut rand::thread_rng()),
    );
    *rate_bias += rate_bias_noise;

    let rate_normal = Normal::new(0.0, 0.01).unwrap();
    let rate_noise = Vector3::new(
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_rate = state.rate() + rate_noise + *rate_bias;

    (noisy_accl, noisy_rate)
}

pub fn gps(state: &State) -> Vector3<f64> {
    // GPS Position
    let pos_normal = Normal::new(0.0, 0.25).unwrap();
    let pos_noise = Vector3::new(
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_pos = state.position() + pos_noise;

    noisy_pos
}
