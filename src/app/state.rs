use na::{DualQuaternion, Quaternion, RealField, SVector, UnitDualQuaternion, UnitQuaternion};
use nalgebra as na;

#[derive(Debug)]
pub struct State<T: RealField + Copy> {
    pub q: UnitDualQuaternion<T>,
    pub eta: DualQuaternion<T>,
}

impl<T: RealField + Copy> State<T> {
    pub fn from_initial_conditions(
        position: &SVector<T, 3>,
        velocity: &SVector<T, 3>,
        roll: &T,
        pitch: &T,
        yaw: &T,
        omega: &SVector<T, 3>,
    ) -> Self {
        let q = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
            *UnitQuaternion::from_euler_angles(*roll, *pitch, *yaw).quaternion(),
            *UnitQuaternion::from_euler_angles(*roll, *pitch, *yaw).quaternion()
                * Quaternion::from_imag(*position)
                * T::from_f32(0.5).unwrap(),
        ));

        let eta = DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(*omega),
            Quaternion::from_imag(velocity + position.cross(&omega)),
        );

        Self { q, eta }
    }

    pub fn rotation(&self) -> UnitQuaternion<T> {
        UnitQuaternion::from_quaternion(self.q.real)
    }

    pub fn position(&self) -> SVector<T, 3> {
        ((self.q.dual * self.q.real.conjugate()) * T::from_f32(2.0).unwrap()).imag()
    }

    pub fn angular_velocity(&self) -> SVector<T, 3> {
        self.eta.real.imag()
    }

    pub fn velocity(&self) -> SVector<T, 3> {
        self.eta.dual.imag() - self.position().cross(&self.angular_velocity())
    }
}
