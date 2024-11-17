use super::plot;
use super::State;
use anyhow::{Error, Result};
use core::mem::MaybeUninit;
use na::{Quaternion, SMatrix, SVector, Vector3};
use nalgebra as na;
use rerun::external::re_log;

trait ConstrainedAlgebra<T, const N: usize> {
    const CONSTRAINED_DIM: usize = N;
    fn inner_difference(&self, rhs: &T) -> SVector<f64, N>;
    fn inner_displacement(&self, rhs: SVector<f64, N>) -> T;
    fn weighted_mean<const M: usize>(x: &[T; M], w: &[f64; M]) -> Result<T>;
}

#[derive(Debug, Clone)]
struct Vector<const N: usize>(SVector<f64, N>);

impl<const N: usize> Vector<N> {}

impl<const N: usize> From<SVector<f64, N>> for Vector<N> {
    fn from(v: SVector<f64, N>) -> Self {
        Self(v)
    }
}

impl<const N: usize> From<[f64; N]> for Vector<N> {
    fn from(v: [f64; N]) -> Self {
        Self(SVector::<f64, N>::from(v))
    }
}

impl<const N: usize> ConstrainedAlgebra<Self, N> for Vector<N> {
    fn inner_difference(&self, rhs: &Self) -> SVector<f64, N> {
        self.0 - rhs.0
    }

    fn inner_displacement<'a>(&self, rhs: SVector<f64, N>) -> Self {
        (self.0 + rhs).into()
    }

    fn weighted_mean<const M: usize>(x: &[Self; M], w: &[f64; M]) -> Result<Self> {
        let mut e_mean = SVector::<f64, N>::zeros();
        for (x, w) in x.iter().zip(w) {
            e_mean += *w * x.0;
        }

        Ok(e_mean.into())
    }
}

#[derive(Debug, Clone)]
struct UnitQuaternion(na::UnitQuaternion<f64>);

impl UnitQuaternion {
    fn new() -> Self {
        Self(na::UnitQuaternion::from_quaternion(Quaternion::new(
            1.0, 0.0, 0.0, 0.0,
        )))
    }

    fn as_vector(&self) -> &SVector<f64, 4> {
        self.0.as_vector()
    }
}

impl From<na::UnitQuaternion<f64>> for UnitQuaternion {
    fn from(q: na::UnitQuaternion<f64>) -> Self {
        Self(q)
    }
}

impl From<SVector<f64, 4>> for UnitQuaternion {
    fn from(v: SVector<f64, 4>) -> Self {
        Self(na::UnitQuaternion::from_quaternion(Quaternion::from(v)))
    }
}

impl From<Quaternion<f64>> for UnitQuaternion {
    fn from(q: Quaternion<f64>) -> Self {
        Self(na::UnitQuaternion::from_quaternion(q))
    }
}

impl ConstrainedAlgebra<Self, 3> for UnitQuaternion {
    fn inner_difference(&self, rhs: &Self) -> SVector<f64, 3> {
        (self.0 * rhs.0.conjugate()).ln().vector().into()
    }

    fn inner_displacement<'a>(&self, rhs: SVector<f64, 3>) -> Self {
        (na::UnitQuaternion::from_quaternion(na::UnitQuaternion::new(rhs).exp()) * self.0).into()
    }

    fn weighted_mean<const M: usize>(x: &[Self; M], w: &[f64; M]) -> Result<Self> {
        // Quaternion weighted mean based on the method presented by Markley et al.
        // REFERENCE: https://doi.org/10.2514/1.28949
        let mut q_accu = SMatrix::<f64, 4, 4>::zeros();
        for (x, w) in x.iter().zip(w) {
            q_accu += *w * (x.as_vector() * x.as_vector().transpose());
        }

        // Find the eigenvector associated with the largest eigenvalue of the accumulator matrix
        // SVD decomp is guaranteed to generate eigenvalues in descending magnitude
        let q_mean = UnitQuaternion::from(
            q_accu
                .svd(true, false)
                .u
                .ok_or(Error::msg("Failed to compute SVD"))?
                .fixed_columns::<1>(0)
                .clone_owned(),
        );

        Ok(q_mean)
    }
}

#[derive(Debug, Clone)]
struct MixedVector<const N: usize> {
    q: UnitQuaternion,
    e: SVector<f64, N>,
}

impl<const N: usize> MixedVector<N> {
    fn default() -> Self {
        Self {
            q: UnitQuaternion::new(),
            e: SVector::<f64, N>::zeros(),
        }
    }
    fn quaternion(&self) -> &UnitQuaternion {
        &self.q
    }

    fn euclidean(&self) -> &SVector<f64, N> {
        &self.e
    }

    fn from_components(q: UnitQuaternion, e: SVector<f64, N>) -> Self {
        Self { q, e }
    }
}

impl<const N: usize> ConstrainedAlgebra<Self, { N + 3 }> for MixedVector<N> {
    fn inner_difference(&self, rhs: &Self) -> SVector<f64, { N + 3 }> {
        let mut d = SVector::<f64, { N + 3 }>::zeros();
        let dq = self.quaternion().inner_difference(rhs.quaternion());
        let de = self.euclidean() - rhs.euclidean();
        d.fixed_rows_mut::<3>(0).copy_from(&dq);
        d.fixed_rows_mut::<N>(3).copy_from(&de);

        d
    }

    fn inner_displacement(&self, rhs: SVector<f64, { N + 3 }>) -> Self {
        Self {
            q: self
                .quaternion()
                .inner_displacement(rhs.fixed_rows::<3>(0).clone_owned()),
            e: self.euclidean() - rhs.fixed_rows::<N>(3),
        }
    }

    fn weighted_mean<const M: usize>(x: &[Self; M], w: &[f64; M]) -> Result<Self> {
        // Quaternion weighted mean based on the method presented by Markley et al.
        // REFERENCE: https://doi.org/10.2514/1.28949
        let mut q_accu = SMatrix::<f64, 4, 4>::zeros();
        let mut e_mean = SVector::<f64, N>::zeros();
        for (x, w) in x.iter().zip(w) {
            q_accu += *w * (x.quaternion().as_vector() * x.quaternion().as_vector().transpose());
            e_mean += *w * x.euclidean();
        }

        // Find the eigenvector associated with the largest eigenvalue of the accumulator matrix
        // SVD decomp is guaranteed to generate eigenvalues in descending magnitude
        let q_mean = UnitQuaternion::from(
            q_accu
                .svd(true, false)
                .u
                .ok_or(Error::msg("Failed to compute SVD"))?
                .fixed_columns::<1>(0)
                .clone_owned(),
        );

        Ok(Self::from_components(q_mean, e_mean))
    }
}

type StateVector = MixedVector<3>;
type StateVectorAugmented = MixedVector<6>;

fn measurement_model(m: &Vector<6>, _u: &(), _dt: &()) -> UnitQuaternion {
    let a = SVector::<f64, 3>::from(m.0.fixed_rows::<3>(0)).normalize();
    let b = SVector::<f64, 3>::from(m.0.fixed_rows::<3>(3)).normalize();

    let a_x = -a[0];
    let a_y = -a[1];
    let a_z = a[2];

    // Switch case based on sign of a_z
    let q_a = if a_z >= 0.0 {
        let w = ((a_z + 1.0) / 2.0).sqrt();
        let ijk = SVector::<f64, 3>::from([
            -a_y / (2.0 * (a_z + 1.0)).sqrt(),
            a_x / (2.0 * (a_z + 1.0)).sqrt(),
            0.0,
        ]);
        Quaternion::from_parts(w, ijk)
    } else {
        let w = -a_y / (2.0 * (1.0 - a_x)).sqrt();
        let ijk = SVector::<f64, 3>::from([
            ((1.0 - a_x) / 2.0).sqrt(),
            0.0,
            a_x / (2.0 * (1.0 - a_x)).sqrt(),
        ]);
        Quaternion::from_parts(w, ijk)
    };

    let e_a = UnitQuaternion::from(q_a);

    let l = e_a.0.to_rotation_matrix().transpose() * b;

    let l_x = l[0];
    let l_y = l[1];
    let l_z = l[2];

    // Switch case based on sign of l_x
    let gamma = l_x.powi(2) + l_y.powi(2);

    let q_m = if l_x >= 0.0 {
        let lambda = (gamma + l_x * gamma.sqrt()).sqrt();
        let w = lambda / (2.0 * gamma).sqrt();
        let ijk = SVector::<f64, 3>::from([0.0, 0.0, l_y / ((2.0 as f64).sqrt() * lambda)]);
        Quaternion::from_parts(w, ijk)
    } else {
        let lambda = (gamma - l_x * gamma.sqrt()).sqrt();
        let w = l_y / ((2.0 as f64).sqrt() * lambda);
        let ijk = SVector::<f64, 3>::from([0.0, 0.0, lambda / (2.0 * gamma).sqrt()]);
        Quaternion::from_parts(w, ijk)
    };

    let e_m = UnitQuaternion::from(q_m);

    UnitQuaternion::from(e_a.0 * e_m.0)
}

/// Observation model.
///
/// Arguments:
///
/// * `x` - Quaternion state vector
///
fn observation_model(x: &StateVector, _u: &(), _dt: &()) -> UnitQuaternion {
    x.quaternion().to_owned()
}

/// Compute the expected out of the process given a set of inputs.
///
/// Arguments:
///
/// * `x` - Augmented state vector [attitude_quaternion_vec_4x1; gyroscope_rate_bias_3x1, gyroscope_measurment_noise_3x1]
/// * `w` - Measured gyroscopic rates [w_x; w_y; w_z] (rad/s)
/// * `dt` - Timestep (s)
///
fn process_model(x: &StateVectorAugmented, w: &SVector<f64, 3>, dt: &f64) -> StateVector {
    // Remove estimated bias and noise with gyroscope measurment model
    let b = x.euclidean().fixed_rows::<3>(0);
    let q = x.euclidean().fixed_rows::<3>(3);
    let w = w - b - q;

    // Quaternion rotation kinematic equation
    let phi = (0.5 * w.norm() * dt).sin() * w / w.norm();
    let a = (0.5 * w.norm() * dt).cos();
    let omega = SMatrix::<f64, 4, 4>::new(
        a, phi[2], -phi[1], phi[0], //
        -phi[2], a, phi[0], phi[1], //
        phi[1], -phi[0], a, phi[2], //
        -phi[0], -phi[1], -phi[2], a,
    );
    let e_k = UnitQuaternion::from(omega * x.quaternion().as_vector());

    StateVector::from_components(e_k, b.clone_owned())
}

/// Compute the unscented transformation of the measurment vector through the observation
/// model.
///
/// Arguments:
///
/// * `f` - Observation model function.
/// * `x` - Measurment vector.
/// * `p_xx` - Covariance matrix of measurement vector
///
fn ut<X, Y, U, T, const N: usize, const M: usize>(
    alpha: f64,
    kappa: f64,
    beta: f64,
    f: fn(&X, &U, &T) -> Y,
    x: &X,
    p_xx: &SMatrix<f64, N, N>,
    u: &U,
    dt: &T,
) -> Result<(Y, SMatrix<f64, M, M>, SMatrix<f64, N, M>)>
where
    X: ConstrainedAlgebra<X, N> + Clone + std::fmt::Debug,
    Y: ConstrainedAlgebra<Y, M> + std::fmt::Debug,
    [(); (2 * N) + 1]:,
{
    let lambda = alpha.powi(2) * (N as f64 + kappa) - N as f64;

    // p_xx_root is used to generate a distribution of sigma points
    let mut p = p_xx.clone();
    let l = loop {
        match p.cholesky() {
            Some(c) => break c.l(),
            None => p += SMatrix::<f64, N, N>::from_diagonal_element(1.0),
        }
    };

    // Create and propagate sigma points and weights
    let mut x_cal: [MaybeUninit<X>; (2 * N) + 1] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut y_cal: [MaybeUninit<Y>; (2 * N) + 1] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut w_m: [MaybeUninit<f64>; (2 * N) + 1] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut w_c: [MaybeUninit<f64>; (2 * N) + 1] = unsafe { MaybeUninit::uninit().assume_init() };

    for (sign, i_offset) in [(-1.0, 1), (1.0, N + 1)] {
        for i in 0..N {
            let sigma_point = x.inner_displacement(
                sign * (N as f64 + lambda).sqrt() * l.fixed_view::<N, 1>(0, i).clone_owned(),
            );
            let m = 0.5 / (N as f64 + lambda);
            let c = 0.5 / (N as f64 + lambda);

            x_cal[i + i_offset].write(sigma_point.clone());
            y_cal[i + i_offset].write(f(&sigma_point, u, dt));
            w_m[i + i_offset].write(m);
            w_c[i + i_offset].write(c);
        }
    }

    x_cal[0].write(x.clone());
    y_cal[0].write(f(&x, u, dt));
    w_m[0].write(lambda / (N as f64 + lambda));
    w_c[0].write(lambda / (N as f64 + lambda) + (1.0 - alpha.powi(2) + beta));

    let x_cal = unsafe { MaybeUninit::array_assume_init(x_cal) };
    let y_cal = unsafe { MaybeUninit::array_assume_init(y_cal) };
    let w_m = unsafe { MaybeUninit::array_assume_init(w_m) };
    let w_c = unsafe { MaybeUninit::array_assume_init(w_c) };

    let mut o = 0.0;
    let mut p = 0.0;
    for (a, b) in w_m.iter().zip(w_c) {
        o += a;
        p += b;
    }

    // Weighted mean
    let y = Y::weighted_mean(&y_cal, &w_m)?;

    // Covariance
    let mut p_yy = SMatrix::<f64, M, M>::zeros();
    let mut p_xy = SMatrix::<f64, N, M>::zeros();

    for ((sigma_point, prop_sigma_point), w) in x_cal.iter().zip(y_cal).zip(w_c) {
        // Quaternion error
        let dy = prop_sigma_point.inner_difference(&y);

        // Cross error
        let dx = sigma_point.inner_difference(x);

        p_yy += w * dy * dy.transpose();
        p_xy += w * dx * dy.transpose();
    }

    Ok((y, p_yy, p_xy))
}

pub struct UkfState {
    q1: SMatrix<f64, 3, 3>,
    q2: SMatrix<f64, 6, 6>,
    r: SMatrix<f64, 6, 6>,
    x_kk1: StateVector,
    p_xx_kk1: SMatrix<f64, 6, 6>,
}

impl UkfState {
    pub fn new() -> Self {
        // Covariance matrix of multiplicative process noise = diag([gyroscope_noise_3x1])
        let q1 = SMatrix::<f64, 3, 3>::from_diagonal(&SVector::<f64, 3>::from([10.0, 10.0, 10.0]));

        // Covariance matrix of additive process noise = diag([quaternion_process_noise_3x1, gyroscope_bias_3x1])
        let q2 = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::from([
            0.00000000001,
            0.00000000001,
            0.00000000001,
            0.001,
            0.001,
            0.001,
        ]));

        // Covariance matrix of measurment noise = diag([accelerometer_noise_3x1, magnetometer_noise_3x1])
        let r = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::from([
            1.0, 1.0, 1.0, 100.0, 100.0, 100.0,
        ]));

        // State vector = [attitude_quaternion_vec_4x1; gyroscope_rate_bias_3x1]
        let x_kk1 = StateVector::default();

        // State covariance matrix = diag([rotation_vector_3x1, gyroscope_bias_3x1])
        let p_xx_kk1 = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::from([
            0.1, 0.1, 0.1, 0.001, 0.001, 0.001,
        ]));

        Self {
            q1,
            q2,
            r,
            x_kk1,
            p_xx_kk1,
        }
    }
}

pub fn ukf(
    rec: &rerun::RecordingStream,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    accl: &Vector3<f64>,
    rate: &Vector3<f64>,
    state: &State,
    ukf_state: &mut UkfState,
    dt: f64,
    t: f64,
) -> Result<State> {
    //
    // Measurement
    //

    // Measurment vector (m/s^2, guass)
    let mag = state.attitude() * Vector3::<f64>::new(1.0, 0.0, 0.0); // TEMPORARY
    let m = Vector::<6>::from([accl[0], accl[1], accl[2], mag[0], mag[1], mag[2]]);

    // Gryoscopic rate vector (rad/s)
    let w = SVector::<f64, 3>::from([rate[0], rate[1], rate[2]]);

    // Compute the attitude and covariance from the accelerometer measurement
    // TODO: Fix the .unwrap()
    let (y_k, r_k, _) = ut(
        1.0,
        0.01,
        2.0,
        measurement_model,
        &m,
        &ukf_state.r,
        &(),
        &(),
    )?;

    //
    // Forecast
    //

    // Augmented state vector = [state_vector_7x1; q1_vector_3x1]
    let mut e = SVector::<f64, 6>::zeros();
    e.fixed_rows_mut::<3>(0)
        .copy_from(ukf_state.x_kk1.euclidean());
    let x_aug = StateVectorAugmented::from_components(ukf_state.x_kk1.quaternion().clone(), e);

    // Augmented state covariance matrix = diag([state_covariance_6x6, multiplacative_process_noise_covariance_3x3])
    let mut p_xx_aug = SMatrix::<f64, 9, 9>::zeros();
    p_xx_aug
        .fixed_view_mut::<6, 6>(0, 0)
        .copy_from(&ukf_state.p_xx_kk1);
    p_xx_aug
        .fixed_view_mut::<3, 3>(6, 6)
        .copy_from(&ukf_state.q1);

    // Predict the state and covariance via an unscented transformation of the augmented state + covariance
    (ukf_state.x_kk1, ukf_state.p_xx_kk1, _) = ut(
        1.0,
        0.01,
        2.0,
        process_model,
        &x_aug,
        &p_xx_aug,
        &w,
        &(dt as f64),
    )?;

    // State covariance update
    ukf_state.p_xx_kk1 += ukf_state.q2;

    // Predict measurement
    let (y_kk1, mut p_yy_kk1, p_xy_kk1) = ut(
        1.0,
        0.01,
        2.0,
        observation_model,
        &ukf_state.x_kk1,
        &ukf_state.p_xx_kk1,
        &(),
        &(),
    )?;

    // Measurment covariance update
    p_yy_kk1 += r_k;

    // Innovation
    let v_k = y_k.inner_difference(&y_kk1);

    //
    // Data assimilation
    //

    // Kalman gain
    // TODO: Fix the .unwrap()
    let k_k = p_xy_kk1 * p_yy_kk1.try_inverse().unwrap();

    // State estimate
    let v_k = y_k.inner_difference(&y_kk1);
    let x_k = ukf_state.x_kk1.inner_displacement(k_k * v_k);

    // Covariance estimate
    let p_xx_k = ukf_state.p_xx_kk1 - k_k * p_yy_kk1 * k_k.transpose();

    // UPDATE HACK
    ukf_state.x_kk1 = x_k;
    ukf_state.p_xx_kk1 = p_xx_k;

    let q = ukf_state.x_kk1.quaternion().0;

    plot::plot_ukf(rec, &q, t)?;

    Ok(state.clone())
}
