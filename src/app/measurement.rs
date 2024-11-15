use super::State;
use na::Vector3;
use nalgebra as na;
use rand_distr::{Distribution, Normal};

pub fn measurment(state: &State) -> State {
    // GPS Position
    let pos_normal = Normal::new(0.0, 0.01).unwrap();
    let pos_noise = Vector3::new(
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
        pos_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_pos = state.position() + pos_noise;

    // GPS Velocity
    let vel_normal = Normal::new(0.0, 0.01).unwrap();
    let vel_noise = Vector3::new(
        vel_normal.sample(&mut rand::thread_rng()),
        vel_normal.sample(&mut rand::thread_rng()),
        vel_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_vel = state.velocity() + vel_noise;

    // Attitude
    // TODO: Change to noisey accel vector
    let (roll, pitch, yaw) = state.attitude().euler_angles();
    let roll_normal = Normal::new(0.0, 0.01).unwrap();
    let noisy_roll = roll + roll_normal.sample(&mut rand::thread_rng());
    let pitch_normal = Normal::new(0.0, 0.01).unwrap();
    let noisy_pitch = pitch + pitch_normal.sample(&mut rand::thread_rng());
    let yaw_normal = Normal::new(0.0, 0.01).unwrap();
    let noisy_yaw = yaw + yaw_normal.sample(&mut rand::thread_rng());

    // Rates
    let rate_normal = Normal::new(0.0, 0.01).unwrap();
    let rate_noise = Vector3::new(
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
        rate_normal.sample(&mut rand::thread_rng()),
    );
    let noisy_rate = state.rate() + rate_noise;

    State::from_initial_conditions(
        &noisy_pos,
        &noisy_vel,
        &noisy_roll,
        &noisy_pitch,
        &noisy_yaw,
        &noisy_rate,
    )
}
