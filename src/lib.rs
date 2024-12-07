#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
use na::{DualQuaternion, Quaternion, SMatrix, SVector, Vector3};
use nalgebra as na;
use pyo3::prelude::*;

mod app;

pub use app::state::State;

#[pyfunction(signature = (accl, rate, mag, ukf_state, dt, pos = None))]
pub fn ukf_py(
    accl: [f64; 3],
    rate: [f64; 3],
    mag: [f64; 3],
    ukf_state: &mut UkfState,
    dt: f64,
    pos: Option<[f64; 3]>,
) -> ([f64; 4], [f64; 3]) {
    let mut ukf_state = app::ukf::UkfState {
        q: ukf_state.q,
        r: ukf_state.r,
        x_kk1: ukf_state.x_kk1,
        p_xx_kk1: ukf_state.p_xx_kk1,
    };
    let state_est = app::ukf::ukf_py(accl, rate, mag, &mut ukf_state, dt, pos);

    (
        (*state_est.attitude().as_vector()).into(),
        state_est.position().into(),
    )
}

#[derive(Debug)]
#[pyclass]
pub struct UkfState {
    pub q: SMatrix<f64, 12, 12>,
    pub r: SMatrix<f64, 9, 9>,
    pub x_kk1: app::ukf::StateVector,
    pub p_xx_kk1: SMatrix<f64, 12, 12>,
}

#[pymethods]
impl UkfState {
    #[new]
    pub fn new(accl: f64) -> Self {
        // Covariance matrix of additive process noise = diag([dual_quat_process_noise_3x1, pos_process_noise_3x1, vel_procces_noise_3x1, gyroscope_bias_3x1])
        let q = SMatrix::<f64, 12, 12>::from_diagonal(&SVector::<f64, 12>::from([
            0.001,
            0.001,
            0.001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.00000000001,
        ]));

        // Covariance matrix of measurment noise = diag([accelerometer_noise_3x1, magnetometer_noise_3x1, pos_noise_3x1])
        let r = SMatrix::<f64, 9, 9>::from_diagonal(&SVector::<f64, 9>::from([
            accl, accl, accl, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1,
        ]));

        // State vector = [real_quaternion_vec_4x1; dual_quaternion_vec_4x1, gyroscope_rate_bias_3x1]
        let mut x_kk1 = app::ukf::StateVector::default();
        x_kk1.q = app::ukf::UnitDualQuaternion::new();

        // State covariance matrix = diag([rotation_vector_3x1, pos_vector_3x1, vel_vector_3x1, gyroscope_bias_3x1])
        let p_xx_kk1 = SMatrix::<f64, 12, 12>::from_diagonal(&SVector::<f64, 12>::from([
            0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        ]));

        Self {
            q,
            r,
            x_kk1,
            p_xx_kk1,
        }
    }
}

#[pymodule]
fn dual_quat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ukf_py, m)?)?;
    m.add_class::<UkfState>()?;
    Ok(())
}
