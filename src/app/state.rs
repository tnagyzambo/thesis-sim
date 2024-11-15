use na::{
    DualQuaternion, Matrix3, Matrix4, Quaternion, UnitDualQuaternion, UnitQuaternion, Vector3,
};
use nalgebra as na;
use re_viewer::external::{eframe, egui, re_log};

const ROOT2: f64 = 1.4142135623730950488016887242096980785696718753769480731766797379;

pub const R1: f64 = 1.41421356237309504;
pub const R2: f64 = 1.41421356237309504;
pub const R3: f64 = 1.41421356237309504;
pub const R4: f64 = 1.41421356237309504;

pub const Q_INVERT: UnitQuaternion<f64> =
    UnitQuaternion::new_unchecked(Quaternion::new(0.0, ROOT2 / 2.0, ROOT2 / 2.0, 0.0));

const J_11: f64 = 0.04338;
const J_22: f64 = 0.04338;
const J_33: f64 = 0.07050;
//const J_11: f64 = 1.0;
//const J_22: f64 = 1.0;
//const J_33: f64 = 1.0;

pub const J: Matrix3<f64> = Matrix3::<f64>::new(
    J_11, 0.0, 0.0, //
    0.0, J_22, 0.0, //
    0.0, 0.0, J_33,
);

pub const J_INV: Matrix3<f64> = Matrix3::<f64>::new(
    1.0 / J_11,
    0.0,
    0.0, //
    0.0,
    1.0 / J_22,
    0.0, //
    0.0,
    0.0,
    1.0 / J_33,
);

pub const M: f64 = 1.27;
pub const M_INV: f64 = 1.0 / M;

const C_T: f64 = 1.5E-5;
const C_M: f64 = 1.9E-7;
const X: f64 = 0.7071067811865475244008;
pub const A: Matrix4<f64> = Matrix4::new(
    C_T,
    C_T,
    C_T,
    C_T, //
    X * R1 * C_T,
    -X * R2 * C_T,
    -X * R3 * C_T,
    X * R4 * C_T, //
    X * R1 * C_T,
    X * R2 * C_T,
    -X * R3 * C_T,
    -X * R4 * C_T, //
    C_M,
    -C_M,
    C_M,
    -C_M,
);

const TC: f64 = 100.0;
pub const MOTOR_A: Matrix4<f64> = Matrix4::new(
    -TC, 0.0, 0.0, 0.0, //
    0.0, -TC, 0.0, 0.0, //
    0.0, 0.0, -TC, 0.0, //
    0.0, 0.0, 0.0, -TC, //
);
pub const MOTOR_B: Matrix4<f64> = Matrix4::new(
    TC, 0.0, 0.0, 0.0, //
    0.0, TC, 0.0, 0.0, //
    0.0, 0.0, TC, 0.0, //
    0.0, 0.0, 0.0, TC, //
);

pub fn dq_exp(q: DualQuaternion<f64>) -> UnitDualQuaternion<f64> {
    UnitDualQuaternion::new_unchecked(DualQuaternion::from_real_and_dual(
        q.real.exp(),
        q.real.exp() * q.dual.clone(),
    ))
}

pub fn dq_ln(q: UnitDualQuaternion<f64>) -> (Vector3<f64>, Vector3<f64>) {
    if q.real.norm() == 0.0 {
        (Vector3::<f64>::zeros(), Vector3::<f64>::zeros())
    } else {
        (q.real.ln().imag(), (q.dual / q.real.norm()).imag())
    }
}

pub fn q_ln(q: UnitQuaternion<f64>) -> Vector3<f64> {
    let q = q.normalize();
    let norm = q.norm();

    if norm == 0.0 {
        Vector3::<f64>::zeros()
    } else {
        (q.imag() / norm) * q.scalar().acos()
    }
}

pub fn q_product(a: Quaternion<f64>, b: Quaternion<f64>) -> Quaternion<f64> {
    Quaternion::from_parts(
        a.scalar() * b.scalar() - a.imag().dot(&b.imag()),
        a.scalar() * b.imag() + b.scalar() * a.imag() + a.imag().cross(&b.imag()),
    )
}

pub fn dq_dot(a: DualQuaternion<f64>, b: DualQuaternion<f64>) -> (f64, f64) {
    (
        a.real.dot(&b.real),
        a.real.dot(&b.dual) + a.dual.dot(&b.real),
    )
}

pub struct Force {
    pub f: Vector3<f64>,
    pub name: String,
}

impl Force {
    pub fn new(f: Vector3<f64>, name: String) -> Self {
        Self { f, name }
    }
}

pub struct Torque {
    pub tau: Vector3<f64>,
    pub name: String,
}

impl Torque {
    pub fn new(tau: Vector3<f64>, name: String) -> Self {
        Self { tau, name }
    }
}

//pub fn cross(self, b: DualQuaternion<T>) -> DualQuaternion<T> {
//    Self((self.clone() * b.clone() - b.conjugate() * self.conjugate()) * 0.5.into())
//}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub q: UnitDualQuaternion<f64>,
    pub xi: DualQuaternion<f64>,
}

impl State {
    pub fn from_initial_conditions(
        position: &Vector3<f64>,
        velocity: &Vector3<f64>,
        roll: &f64,
        pitch: &f64,
        yaw: &f64,
        omega: &Vector3<f64>,
    ) -> Self {
        let rotation = Q_INVERT * UnitQuaternion::from_euler_angles(*roll, *pitch, *yaw);
        let velocity_body = rotation.conjugate() * velocity;
        let position_body = rotation.conjugate() * position;

        // Inital pose
        let q = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
            *rotation,
            q_product(*rotation.quaternion(), Quaternion::from_imag(position_body)) * 0.5,
        ));

        // Initial twist
        let xi = DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(*omega),
            Quaternion::from_imag(velocity_body), // Omit omega.cross(position_body) since we want to apply an initial inertial velocity
        );

        Self { q, xi }
    }

    pub fn attitude(&self) -> UnitQuaternion<f64> {
        Q_INVERT.conjugate() * UnitQuaternion::new_unchecked(self.q.real)
    }

    pub fn rotation(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::new_unchecked(self.q.real)
    }

    pub fn rate(&self) -> Vector3<f64> {
        self.xi.real.imag()
    }

    pub fn position_body(&self) -> Vector3<f64> {
        (self.q.real.conjugate() * self.q.dual * 2.0).imag()
    }

    pub fn velocity_body(&self) -> Vector3<f64> {
        self.xi.dual.imag() - self.xi.real.imag().cross(&self.position_body())
    }

    pub fn position(&self) -> Vector3<f64> {
        (self.q.dual * self.q.real.conjugate() * 2.0).imag()
    }

    pub fn velocity(&self) -> Vector3<f64> {
        self.rotation().conjugate() * self.velocity_body()
    }

    pub fn compute_wrench(&self, tau: &[Vector3<f64>], f: &[Vector3<f64>]) -> DualQuaternion<f64> {
        let tau: Vector3<f64> = tau.iter().sum();
        let f: Vector3<f64> = f.iter().sum();

        let a = -J_INV * (self.rate().cross(&(J * self.rate())));
        let big_f = DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(a),
            Quaternion::from_imag(
                a.cross(&self.position_body()) + self.rate().cross(&self.velocity_body()),
            ),
        );

        let u = DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(J_INV * tau),
            Quaternion::from_imag(J_INV * tau.cross(&self.position_body()) + M_INV * f),
        );

        big_f + u
    }
}
