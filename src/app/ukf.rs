use super::State;
use na::Vector3;
use nalgebra as na;

pub fn ukf(
    rec: &rerun::RecordingStream,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    accl: &Vector3<f64>,
    rate: &Vector3<f64>,
    state: &State,
) -> State {
    state.clone()
}
