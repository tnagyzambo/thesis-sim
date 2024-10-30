use na::{DualQuaternion, Quaternion, UnitDualQuaternion, UnitQuaternion, Vector3};
use nalgebra as na;

const ROTOR_ARM1: Vector3<f64> = Vector3::<f64>::new(0.1, 0.1, 0.0);
const ROTOR_ARM2: Vector3<f64> = Vector3::<f64>::new(0.1, -0.1, 0.0);
const ROTOR_ARM3: Vector3<f64> = Vector3::<f64>::new(-0.1, 0.1, 0.0);
const ROTOR_ARM4: Vector3<f64> = Vector3::<f64>::new(-0.1, -0.1, 0.0);

pub fn dq_exp(q: DualQuaternion<f64>) -> UnitDualQuaternion<f64> {
    UnitDualQuaternion::new_unchecked(DualQuaternion::from_real_and_dual(
        q.real.exp(),
        q.real.exp() * q.dual.clone(),
    ))
}

//pub fn cross(self, b: DualQuaternion<T>) -> DualQuaternion<T> {
//    Self((self.clone() * b.clone() - b.conjugate() * self.conjugate()) * 0.5.into())
//}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub q: UnitDualQuaternion<f64>,
    pub eta: DualQuaternion<f64>,
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
        // Inital pose
        let q = UnitDualQuaternion::from_parts(
            (*position).into(),
            UnitQuaternion::from_euler_angles(*roll, *pitch, *yaw),
        );

        // Initial twist
        let eta = DualQuaternion::from_real_and_dual(
            Quaternion::from_imag(*omega),
            Quaternion::from_imag(q.rotation().conjugate() * velocity),
        );

        Self { q, eta }
    }

    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.q.rotation()
    }
    pub fn angular_velocity_body(&self) -> Vector3<f64> {
        self.eta.real.imag()
    }

    pub fn position_body(&self) -> Vector3<f64> {
        self.q.translation().vector
    }

    pub fn velocity_body(&self) -> Vector3<f64> {
        self.eta.dual.imag() - self.angular_velocity_body().cross(&self.position_body())
    }

    pub fn position(&self) -> Vector3<f64> {
        (self.q.dual * self.q.real.conjugate() * 2.0).imag()
    }

    pub fn velocity(&self) -> Vector3<f64> {
        let eta = (&self.q * self.eta) * self.q.conjugate();
        eta.dual.imag() - self.position().cross(&eta.real.imag())
    }

    pub fn log(&self, rec: &rerun::RecordingStream, t: f64) {
        // Set timestep
        rec.set_time_seconds("sim_time", t);

        // Calcs
        let q = self.rotation(); // Body attitude
        let r = self.position(); // Body position expressed in body frame

        // Plot drone
        rec.log(
            "3d/drone",
            &rerun::InstancePoses3D::new()
                .with_quaternions([q.into_rerun_quaternion()])
                .with_translations([r.into_rerun_vec3d()]),
        )
        .unwrap();

        // Plot body frame
        let x = q * Vector3::<f64>::x_axis().into_inner() * 0.1; // Body frame x-axis expressed in inertial frame
        let y = q * Vector3::<f64>::y_axis().into_inner() * 0.1; // Body frame y-axis expressed in inertial frame
        let z = q * Vector3::<f64>::z_axis().into_inner() * 0.1; // Body frame z-axis expressed in inertial frame
        rec.log(
            "3d/body_frame",
            &rerun::Arrows3D::from_vectors([
                x.into_rerun_vec3d(),
                y.into_rerun_vec3d(),
                z.into_rerun_vec3d(),
            ])
            .with_origins([r.into_rerun_vec3d()])
            .with_colors([
                rerun::Color::from_rgb(255, 0, 0),
                rerun::Color::from_rgb(0, 255, 0),
                rerun::Color::from_rgb(0, 0, 255),
            ]),
        )
        .unwrap();

        // Plot rotor forces
        let rotor_force1 = (q * Vector3::<f64>::z_axis().into_inner()) * 0.1;
        let rotor_force2 = (q * Vector3::<f64>::z_axis().into_inner()) * 0.1;
        let rotor_force3 = (q * Vector3::<f64>::z_axis().into_inner()) * 0.1;
        let rotor_force4 = (q * Vector3::<f64>::z_axis().into_inner()) * 0.1;
        rec.log(
            "3d/rotor_force",
            &rerun::Arrows3D::from_vectors([
                rotor_force1.into_rerun_vec3d(),
                rotor_force2.into_rerun_vec3d(),
                rotor_force3.into_rerun_vec3d(),
                rotor_force4.into_rerun_vec3d(),
            ])
            .with_origins([
                (r + q * ROTOR_ARM1).into_rerun_vec3d(),
                (r + q * ROTOR_ARM2).into_rerun_vec3d(),
                (r + q * ROTOR_ARM3).into_rerun_vec3d(),
                (r + q * ROTOR_ARM4).into_rerun_vec3d(),
            ])
            .with_colors([rerun::Color::from_rgb(255, 100, 100)]),
        )
        .unwrap();

        // Log attitude
        let (roll, pitch, yaw) = q.euler_angles();
        rec.log("attitude/roll", &rerun::Scalar::new(roll)).unwrap();
        rec.log("attitude/pitch", &rerun::Scalar::new(pitch))
            .unwrap();
        rec.log("attitude/yaw", &rerun::Scalar::new(yaw)).unwrap();

        // Log angular velocity (body frame)
        let rate = self.angular_velocity_body();
        rec.log("omega/roll", &rerun::Scalar::new(rate[0])).unwrap();
        rec.log("omega/pitch", &rerun::Scalar::new(rate[1]))
            .unwrap();
        rec.log("omega/yaw", &rerun::Scalar::new(rate[2])).unwrap();

        // Log position (body frame)
        let r_body = self.position_body();
        rec.log("position/body/x", &rerun::Scalar::new(r_body[0]))
            .unwrap();
        rec.log("position/body/y", &rerun::Scalar::new(r_body[1]))
            .unwrap();
        rec.log("position/body/z", &rerun::Scalar::new(r_body[2]))
            .unwrap();

        // Log position (inertial frame)
        rec.log("position/inertial/x", &rerun::Scalar::new(r[0]))
            .unwrap();
        rec.log("position/inertial/y", &rerun::Scalar::new(r[1]))
            .unwrap();
        rec.log("position/inertial/z", &rerun::Scalar::new(r[2]))
            .unwrap();

        // Log velocity (body frame)
        let v = self.velocity_body();
        rec.log("velocity/body/x", &rerun::Scalar::new(v[0]))
            .unwrap();
        rec.log("velocity/body/y", &rerun::Scalar::new(v[1]))
            .unwrap();
        rec.log("velocity/body/z", &rerun::Scalar::new(v[2]))
            .unwrap();

        // Log velocity (inertial frame)
        let v = self.velocity();
        rec.log("velocity/inertial/x", &rerun::Scalar::new(v[0]))
            .unwrap();
        rec.log("velocity/inertial/y", &rerun::Scalar::new(v[1]))
            .unwrap();
        rec.log("velocity/inertial/z", &rerun::Scalar::new(v[2]))
            .unwrap();
    }
}

trait IntoRerunQuaternion {
    fn into_rerun_quaternion(&self) -> rerun::Quaternion;
}

impl IntoRerunQuaternion for Quaternion<f64> {
    fn into_rerun_quaternion(&self) -> rerun::Quaternion {
        rerun::Quaternion::from_xyzw(std::array::from_fn(|i| self[i] as f32))
    }
}

trait IntoRerunVec3D {
    fn into_rerun_vec3d(&self) -> rerun::Vec3D;
}

impl IntoRerunVec3D for Vector3<f64> {
    fn into_rerun_vec3d(&self) -> rerun::Vec3D {
        rerun::Vec3D::from(std::array::from_fn(|i| self[i] as f32))
    }
}
