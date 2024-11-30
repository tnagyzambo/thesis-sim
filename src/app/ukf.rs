use super::plot;
use super::state::Q_INVERT;
use super::State;
use anyhow::{anyhow, Error, Result};
use core::mem::MaybeUninit;
use na::{DualQuaternion, Quaternion, SMatrix, SVector, Vector3};
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

impl<const N: usize> Vector<N> {
    fn new() -> Self {
        Self(SVector::<f64, N>::zeros())
    }
}

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
pub struct UnitDualQuaternion(na::UnitDualQuaternion<f64>);

impl UnitDualQuaternion {
    fn new() -> Self {
        Self(na::UnitDualQuaternion::new_normalize(
            DualQuaternion::from_real_and_dual(Quaternion::identity(), Quaternion::identity()),
        ))
    }
}

impl From<na::UnitDualQuaternion<f64>> for UnitDualQuaternion {
    fn from(q: na::UnitDualQuaternion<f64>) -> Self {
        Self(q)
    }
}

impl From<SVector<f64, 8>> for UnitDualQuaternion {
    fn from(v: SVector<f64, 8>) -> Self {
        Self(na::UnitDualQuaternion::new_normalize(
            DualQuaternion::from_real_and_dual(
                Quaternion::from_vector(v.fixed_rows::<4>(0).clone_owned()),
                Quaternion::from_vector(v.fixed_rows::<4>(4).clone_owned()),
            ),
        ))
    }
}

impl From<DualQuaternion<f64>> for UnitDualQuaternion {
    fn from(q: DualQuaternion<f64>) -> Self {
        Self(na::UnitDualQuaternion::new_normalize(q))
    }
}

impl ConstrainedAlgebra<Self, 6> for UnitDualQuaternion {
    //  super::state::dq_ln(self.0 * rhs.0.conjugate())
    fn inner_difference(&self, rhs: &Self) -> SVector<f64, 6> {
        let trans_lhs = (self.0.real.conjugate() * self.0.dual * 2.0).imag();
        let trans_rhs = (rhs.0.real.conjugate() * rhs.0.dual * 2.0).imag();

        na::stack![UnitQuaternion::from(self.0.real).inner_difference(&UnitQuaternion::from(rhs.0.real));
             na::UnitQuaternion::from_quaternion(rhs.0.real).conjugate() * (trans_lhs - trans_rhs)]
    }

    //         (super::dq_exp(na::DualQuaternion::from_real_and_dual(
    //         na::Quaternion::from_imag(rhs.fixed_rows::<3>(0).into()),
    //         na::Quaternion::from_imag(rhs.fixed_rows::<3>(3).into()),
    //     )) * self.0).into()

    fn inner_displacement<'a>(&self, rhs: SVector<f64, 6>) -> Self {
        let rotation = na::UnitQuaternion::new(rhs.fixed_rows::<3>(0)).exp() * self.0.real;
        let trans_lhs = (self.0.real.conjugate() * self.0.dual * 2.0).imag();

        na::DualQuaternion::from_real_and_dual(
            rotation,
            super::state::q_product(
                rotation,
                na::Quaternion::from_imag(
                    trans_lhs
                        + na::UnitQuaternion::from_quaternion(self.0.real) * rhs.fixed_rows::<3>(3),
                ) * 0.5,
            ),
        )
        .into()
    }

    fn weighted_mean<const M: usize>(x: &[Self; M], w: &[f64; M]) -> Result<Self> {
        let rotations: [UnitQuaternion; M] =
            core::array::from_fn(|i| UnitQuaternion::from(x[i].0.real));
        let translations: [Vector<3>; M] =
            core::array::from_fn(|i| (x[i].0.real.conjugate() * x[i].0.dual * 2.0).imag().into());

        let rotation_mean = UnitQuaternion::weighted_mean(&rotations, w)?;
        let translation_mean = Vector::<3>::weighted_mean(&translations, w)?;

        let q_mean = UnitDualQuaternion::from(na::DualQuaternion::from_real_and_dual(
            *rotation_mean.0.quaternion(),
            super::state::q_product(
                *rotation_mean.0.quaternion(),
                na::Quaternion::from_imag(translation_mean.0),
            ) * 0.5,
        ));

        Ok(q_mean)
    }
}

#[derive(Debug, Clone)]
struct MixedVector<const N: usize> {
    q: UnitDualQuaternion,
    e: Vector<N>,
}

impl<const N: usize> MixedVector<N> {
    fn default() -> Self {
        Self {
            q: UnitDualQuaternion::new(),
            e: Vector::<N>::new(),
        }
    }
    fn dual_quaternion(&self) -> &UnitDualQuaternion {
        &self.q
    }

    fn euclidean(&self) -> &Vector<N> {
        &self.e
    }

    fn from_components(q: UnitDualQuaternion, e: Vector<N>) -> Self {
        Self { q, e }
    }
}

impl<const N: usize> ConstrainedAlgebra<Self, { N + 6 }> for MixedVector<N> {
    fn inner_difference(&self, rhs: &Self) -> SVector<f64, { N + 6 }> {
        let mut d = SVector::<f64, { N + 6 }>::zeros();
        let dq = self
            .dual_quaternion()
            .inner_difference(rhs.dual_quaternion());
        let de = self.euclidean().0 - rhs.euclidean().0;
        d.fixed_rows_mut::<6>(0).copy_from(&dq);
        d.fixed_rows_mut::<N>(6).copy_from(&de);

        d
    }

    fn inner_displacement(&self, rhs: SVector<f64, { N + 6 }>) -> Self {
        Self {
            q: self
                .dual_quaternion()
                .inner_displacement(rhs.fixed_rows::<6>(0).clone_owned()),
            e: Vector::from(self.euclidean().0 - rhs.fixed_rows::<N>(6)),
        }
    }

    fn weighted_mean<const M: usize>(x: &[Self; M], w: &[f64; M]) -> Result<Self> {
        let q: [UnitDualQuaternion; M] = core::array::from_fn(|i| x[i].dual_quaternion().clone());
        let e: [Vector<N>; M] = core::array::from_fn(|i| x[i].euclidean().clone());

        let q_mean = UnitDualQuaternion::weighted_mean(&q, w)?;
        let e_mean = Vector::<N>::weighted_mean(&e, w)?;

        Ok(Self::from_components(q_mean, e_mean))
    }
}

type StateVector = MixedVector<6>;

fn measurement_model(m: &Vector<9>, _u: &(), _dt: &()) -> UnitDualQuaternion {
    let a = SVector::<f64, 3>::from(m.0.fixed_rows::<3>(0)).normalize();
    let b = SVector::<f64, 3>::from(m.0.fixed_rows::<3>(3)).normalize();
    let r = SVector::<f64, 3>::from(m.0.fixed_rows::<3>(6));

    let a_x = -a[0];
    let a_y = a[1];
    let a_z = a[2];

    // Switch case based on sign of a_z
    let q_a = if a_z >= 0.0 {
        let lambda = ((a_z + 1.0) / 2.0).sqrt();
        let w = lambda;
        let ijk = SVector::<f64, 3>::from([-a_y / (2.0 * lambda), a_x / (2.0 * lambda), 0.0]);
        Quaternion::from_parts(w, ijk)
    } else {
        let lambda = (2.0 * (1.0 - a_z)).sqrt();
        let w = -a_y / (2.0 * lambda);
        let ijk = SVector::<f64, 3>::from([lambda, 0.0, a_x / (2.0 * lambda)]);
        Quaternion::from_parts(w, ijk)
    };

    let e_a = UnitQuaternion::from(q_a).0;

    let l = e_a * b;

    let l_x = l[0];
    let l_y = l[1];

    // Switch case based on sign of l_x
    let gamma = l_x.powi(2) + l_y.powi(2);

    let q_m = if l_x >= 0.0 {
        let lambda = (gamma + l_x * gamma.sqrt()).sqrt();
        let w = lambda / (2.0 * gamma).sqrt();
        let ijk = SVector::<f64, 3>::from([0.0, 0.0, lambda / ((2.0 as f64).sqrt() * gamma)]);
        Quaternion::from_parts(w, ijk)
    } else {
        let lambda = (gamma - l_x * gamma.sqrt()).sqrt();
        let w = l_y / ((2.0 as f64).sqrt() * lambda);
        let ijk = SVector::<f64, 3>::from([0.0, 0.0, lambda / (2.0 * gamma).sqrt()]);
        Quaternion::from_parts(w, ijk)
    };

    let e_m = UnitQuaternion::from(q_m).0;

    let rotation = na::UnitQuaternion::new_normalize(Quaternion::new(
        -(2.0_f64).sqrt() / 2.0,
        0.0,
        0.0,
        (2.0_f64).sqrt() / 2.0,
    )) * na::UnitQuaternion::new_normalize(super::state::q_product(
        *e_a.quaternion(),
        *e_m.quaternion(),
    )) * Q_INVERT;
    //let rotation = na::UnitQuaternion::from(e_a) * Q_INVERT;

    let r_body = rotation * r;

    UnitDualQuaternion::from(na::DualQuaternion::from_real_and_dual(
        *rotation,
        super::state::q_product(*rotation, Quaternion::from_imag(r_body)) * 0.5,
    ))
}

/// Observation model.
///
/// Arguments:
///
/// * `x` - Quaternion state vector
///
fn observation_model(x: &StateVector, _u: &(), _dt: &()) -> UnitDualQuaternion {
    x.dual_quaternion().to_owned()
}

/// Compute the expected out of the process given a set of inputs.
///
/// Arguments:
///
/// * `x` - Augmented state vector [attitude_quaternion_vec_4x1; vel_3x1, gyroscope_rate_bias_3x1, gyroscope_measurment_noise_3x1]
/// * `w` - Measured gyroscopic rates [w_x; w_y; w_z] (rad/s)
/// * `dt` - Timestep (s)
///
fn process_model(x: &StateVector, d: &SVector<f64, 6>, dt: &f64) -> StateVector {
    let w = d.fixed_rows::<3>(0);
    let a = d.fixed_rows::<3>(3);

    // Remove estimated bias and noise with gyroscope measurment model
    let rate_bias = x.euclidean().0.fixed_rows::<3>(3);
    let w = w - rate_bias;

    let v = x.euclidean().0.fixed_rows::<3>(0);
    let g = SVector::<f64, 3>::new(0.0, 0.0, -9.81);
    let v = v + (a + g) * *dt;

    let position = (super::state::q_product(
        x.dual_quaternion().0.real.conjugate(),
        x.dual_quaternion().0.dual,
    ) * 2.0)
        .imag();

    let xi = na::DualQuaternion::from_real_and_dual(
        na::Quaternion::from_imag(w),
        na::Quaternion::from_imag(v + w.cross(&position)),
    );

    let e_k = x.dual_quaternion().0 * super::state::dq_exp(0.5 * dt * xi);

    StateVector::from_components(
        e_k.into(),
        Vector::from(SVector::<f64, 6>::from([
            v[0],
            v[1],
            v[2],
            rate_bias[0],
            rate_bias[1],
            rate_bias[2],
        ])),
    )
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
    let p = p_xx.clone();
    let l = match p.cholesky() {
        Some(c) => c.l(),
        None => (p + SMatrix::<f64, N, N>::identity()) / 2.0,
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

    // let mut o = 0.0;
    // let mut p = 0.0;
    // for (a, b) in w_m.iter().zip(w_c) {
    //     o += a;
    //     p += b;
    // }

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

#[derive(Debug)]
pub struct UkfState {
    q: SMatrix<f64, 12, 12>,
    r: SMatrix<f64, 9, 9>,
    x_kk1: StateVector,
    p_xx_kk1: SMatrix<f64, 12, 12>,
}

impl UkfState {
    pub fn new() -> Self {
        // Covariance matrix of additive process noise = diag([dual_quat_process_noise_3x1, pos_process_noise_3x1, vel_procces_noise_3x1, gyroscope_bias_3x1])
        let q = SMatrix::<f64, 12, 12>::from_diagonal(&SVector::<f64, 12>::from([
            0.00000000001,
            0.00000000001,
            0.00000000001,
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
            0.000015, 0.000015, 0.000015, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
        ]));

        // State vector = [real_quaternion_vec_4x1; dual_quaternion_vec_4x1, gyroscope_rate_bias_3x1]
        let x_kk1 = StateVector::default();

        // State covariance matrix = diag([rotation_vector_3x1, pos_vector_3x1, vel_vector_3x1, gyroscope_bias_3x1])
        let p_xx_kk1 = SMatrix::<f64, 12, 12>::from_diagonal(&SVector::<f64, 12>::from([
            0.01, 0.01, 0.01, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001,
        ]));

        Self {
            q,
            r,
            x_kk1,
            p_xx_kk1,
        }
    }
}

pub fn ukf(
    rec: &rerun::RecordingStream,
    pos: &Option<Vector3<f64>>,
    accl: &Vector3<f64>,
    rate: &Vector3<f64>,
    state: &State,
    ukf_state: &mut UkfState,
    dt: f64,
    t: f64,
) -> Result<State> {
    let pos = match pos {
        Some(pos) => *pos,
        None => (super::state::q_product(
            ukf_state.x_kk1.dual_quaternion().0.dual,
            ukf_state.x_kk1.dual_quaternion().0.real.conjugate(),
        ) * 2.0)
            .imag(),
    };

    //
    // Measurement
    //

    // Measurment vector (m/s^2, guass)
    let mag = state.attitude() * Vector3::<f64>::new(10.0, 0.0, 0.0); // TEMPORARY
    let m = Vector::<9>::from([
        accl[0], accl[1], accl[2], mag[0], mag[1], mag[2], pos[0], pos[1], pos[2],
    ]);

    // Gryoscopic rate vector (rad/s)
    let d = SVector::<f64, 6>::from([rate[0], rate[1], rate[2], accl[0], accl[1], accl[2]]);

    // Compute the attitude and covariance from the accelerometer measurement
    // TODO: Fix the .unwrap()
    re_log::info!("meas");
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

    // Measurment covariance update

    //
    // Forecast
    //

    // Gryoscopic rate vector (rad/s)
    let d = SVector::<f64, 6>::from([rate[0], rate[1], rate[2], accl[0], accl[1], accl[2]]);

    // Predict the state and covariance via an unscented transformation of the augmented state + covariance
    re_log::info!("proc {:?}", ukf_state);
    (ukf_state.x_kk1, ukf_state.p_xx_kk1, _) = ut(
        1.0,
        0.01,
        2.0,
        process_model,
        &ukf_state.x_kk1,
        &ukf_state.p_xx_kk1,
        &d,
        &(dt as f64),
    )?;

    // State covariance update
    ukf_state.p_xx_kk1 += ukf_state.q;

    // Predict measurement
    re_log::info!("obv");
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

    //
    // Data assimilation
    //

    p_yy_kk1 += r_k;

    // Kalman gain
    let p_yy_kk1_inv = p_yy_kk1
        .try_inverse()
        .ok_or_else(|| anyhow!("Inverse Failed"));
    let k_k = p_xy_kk1 * p_yy_kk1_inv?;

    // State estimate
    let v_k = y_k.inner_difference(&y_kk1);
    let x_k = ukf_state.x_kk1.inner_displacement(k_k * v_k);

    // Covariance estimate
    let p_xx_k = ukf_state.p_xx_kk1 - k_k * p_yy_kk1 * k_k.transpose();

    // UPDATE
    ukf_state.x_kk1 = x_k;
    ukf_state.p_xx_kk1 = p_xx_k;

    let q = na::UnitQuaternion::from_quaternion(ukf_state.x_kk1.dual_quaternion().0.real);
    let r = (ukf_state.x_kk1.dual_quaternion().0.real.conjugate()
        * ukf_state.x_kk1.dual_quaternion().0.dual
        * 2.0)
        .imag();
    let rate_bias = SVector::<f64, 3>::zeros() + ukf_state.x_kk1.euclidean().0.fixed_rows::<3>(3);
    let rate = rate - rate_bias;
    let vel = ukf_state.x_kk1.euclidean().0.fixed_rows::<3>(0);
    plot::plot_ukf(rec, &q, &r, &rate_bias, t)?;

    let rate = rate - ukf_state.x_kk1.euclidean().0.fixed_rows::<3>(3);
    let state_est = State {
        q: ukf_state.x_kk1.dual_quaternion().0.clone(),
        xi: DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(rate),
            Quaternion::from_imag(vel + rate.cross(&pos)),
        ),
    };

    Ok(state_est)
}
