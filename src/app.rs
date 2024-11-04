use na::{
    AbstractRotation, DualQuaternion, Matrix3, Matrix4, Quaternion, UnitDualQuaternion,
    UnitQuaternion, UnitVector3, Vector3, Vector4,
};
use nalgebra as na;
use re_viewer::external::{eframe, egui, re_log};
mod state;

use state::{dq_exp, q_ln, State, J, M};

pub struct App {
    rerun_app: re_viewer::App,
    duration: f64,
    dt: f64,
    roll: f64,
    pitch: f64,
    yaw: f64,
    omega: Vector3<f64>,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
}

impl App {
    pub fn new(rerun_app: re_viewer::App) -> Self {
        Self {
            rerun_app,
            duration: 10.0,
            dt: 0.01,
            roll: 0.01,
            pitch: 0.0,
            yaw: 0.0,
            omega: Vector3::<f64>::zeros(),
            position: Vector3::<f64>::new(0.0, 0.0, 1.0),
            velocity: Vector3::<f64>::zeros(),
        }
    }

    fn simulate(&self, _ctx: egui::Context) {
        let rec = re_sdk::RecordingStreamBuilder::new("Simulator")
            .spawn()
            .unwrap();

        let dt = self.dt;
        let n = (self.duration / dt).round() as u32;

        let mut state = State::from_initial_conditions(
            &self.position,
            &self.velocity,
            &self.roll,
            &self.pitch,
            &self.yaw,
            &self.omega,
        );

        rec.log_static(
            "3d/world/drone",
            &rerun::Asset3D::from_file("assets/drone.stl").unwrap(),
        )
        .unwrap();
        rec.log_static(
            "3d/body/drone",
            &rerun::Asset3D::from_file("assets/drone.stl").unwrap(),
        )
        .unwrap();
        rec.log_static(
            "3d/world/interial_frame",
            &rerun::Arrows3D::from_vectors(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .with_origins(&[[0.0, 0.0, 0.0]])
                .with_colors([
                    rerun::Color::from_rgb(255, 0, 0),
                    rerun::Color::from_rgb(0, 255, 0),
                    rerun::Color::from_rgb(0, 0, 255),
                ]),
        )
        .unwrap();
        rec.log_static(
            "3d/body/interial_frame",
            &rerun::Arrows3D::from_vectors(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .with_origins(&[[0.0, 0.0, 0.0]])
                .with_colors([
                    rerun::Color::from_rgb(255, 0, 0),
                    rerun::Color::from_rgb(0, 255, 0),
                    rerun::Color::from_rgb(0, 0, 255),
                ]),
        )
        .unwrap();
        rec.log_static(
            "attitude/roll",
            &rerun::SeriesLine::new().with_color([255, 0, 0]),
        )
        .unwrap();
        rec.log_static(
            "attitude/roll_t",
            &rerun::SeriesLine::new().with_color([200, 0, 0]),
        )
        .unwrap();
        rec.log_static(
            "attitude/pitch",
            &rerun::SeriesLine::new().with_color([0, 255, 0]),
        )
        .unwrap();
        rec.log_static(
            "attitude/pitch_t",
            &rerun::SeriesLine::new().with_color([0, 200, 0]),
        )
        .unwrap();
        rec.log_static(
            "attitude/yaw",
            &rerun::SeriesLine::new().with_color([0, 0, 255]),
        )
        .unwrap();
        rec.log_static(
            "attitude/yaw_t",
            &rerun::SeriesLine::new().with_color([0, 0, 200]),
        )
        .unwrap();

        let k_q = 50.0;
        let k_w = 6.0;
        let k_p = 1.0;
        let k_d = 2.0;
        let k_1 = 0.33;
        let k_2 = 0.2;

        let mut w_t_prev = Vector3::<f64>::zeros();
        let mut w_body_prev = self.omega;

        for i in 0..n {
            let t = i as f64 * dt;
            let a = -J.try_inverse().unwrap()
                * (state
                    .angular_velocity_body()
                    .cross(&(J * state.angular_velocity_body())));

            let wdot_body = (state.angular_velocity_body() - w_body_prev) / dt;
            w_body_prev = state.angular_velocity_body();

            let f = DualQuaternion::from_real_and_dual(
                Quaternion::from_imag(a),
                Quaternion::from_imag(
                    a.cross(&state.position_body())
                        + state.angular_velocity_body().cross(&state.velocity_body()),
                ),
            );

            let f_coriolis = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    -2.0 * M * state.angular_velocity_body().cross(&state.velocity_body()),
                ),
            );

            let f_centrifugal = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    -M * state
                        .angular_velocity_body()
                        .cross(&state.angular_velocity_body().cross(&state.position_body())),
                ),
            );

            let f_euler = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(-M * wdot_body.cross(&state.position_body())),
            );

            let f_g = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    M * (state.rotation().conjugate() * Vector3::<f64>::new(0.0, 0.0, -9.81)),
                ),
            );

            // POS TARGETS
            let p_t = Vector3::<f64>::new(0.0, 0.0, 1.0);
            let pdot_t = Vector3::<f64>::zeros();
            let pddot_t = Vector3::<f64>::zeros();
            //let c1 = 5.0;
            //let c2 = 0.6;
            //let c3 = 10.0;
            //let c4 = 0.5;
            //let c5 = 0.3;
            //let p_t = Vector3::new(
            //    c1 * (c2 * t).sin(),
            //    c1 * (c2 * t).cos(),
            //    c3 - c4 * (c5 * t).sin(),
            //);
            //let pdot_t = Vector3::new(
            //    c1 * c2 * (c2 * t).cos(),
            //    -c1 * c2 * (c2 * t).sin(),
            //    -c4 * c5 * (c5 * t).cos(),
            //);
            //let pddot_t = Vector3::new(
            //    -c1 * c2.powi(2) * (c2 * t).sin(),
            //    -c1 * c2.powi(2) * (c2 * t).cos(),
            //    c4 * c5.powi(2) * (c5 * t).sin(),
            //);

            // GUIDANCE
            let e_n = p_t - state.position();
            let edot_n = pdot_t - state.velocity();

            let e_d = Vector3::<f64>::new(0.0, 0.0, e_n.norm());
            let q_t = if e_d.cross(&e_n).norm() == 0.0 {
                UnitQuaternion::<f64>::identity()
            } else {
                let theta = k_1 * (k_2 * e_n.norm()).atan();
                let axis = e_d.cross(&e_n).normalize();
                UnitQuaternion::new_normalize(Quaternion::from_parts(
                    (theta / 2.0).cos(),
                    axis * (theta / 2.0).sin(),
                ))
            };

            let skew = if e_n.norm() == 0.0 {
                Matrix3::<f64>::zeros()
            } else {
                Matrix3::<f64>::new(
                    0.0,
                    1.0 / e_n.norm(),
                    0.0, //
                    -1.0 / e_n.norm(),
                    0.0,
                    0.0, //
                    0.0,
                    0.0,
                    0.0, //
                )
            };

            let w_t = skew * -(q_t.conjugate() * edot_n);

            let wdot_t = (w_t - w_t_prev) / dt;
            w_t_prev = w_t;

            // ATTITUDE CONTROLLER
            let q_e = q_t.conjugate() * state.rotation();
            let w_e = state.angular_velocity_body() - (q_e.conjugate() * w_t);
            let wdot_e = state.rotation() * (q_t.conjugate() * wdot_t);

            let tau_u = state
                .angular_velocity_body()
                .cross(&(J * state.angular_velocity_body()))
                + J * wdot_e
                - k_q * q_e.imag()
                - k_w * w_e;

            // TRANSLATIONAL CONTROLLER
            let p_e = p_t - state.position();
            let pdot_e = pdot_t - state.velocity();

            let f_thrust =
                -M * pddot_t - Vector3::new(0.0, 0.0, M * -9.81) + k_p * p_e + k_d * pdot_e;

            let f_u = if f_thrust[2] > 30.0 {
                30.0
            } else if f_thrust[2] < -30.0 {
                -0.0
            } else {
                f_thrust[2]
            };

            // INPUT
            let u = DualQuaternion::from_real_and_dual(
                Quaternion::from_imag(Vector3::<f64>::new(tau_u[0], tau_u[1], tau_u[2])),
                Quaternion::from_imag(Vector3::<f64>::new(0.0, 0.0, f_u)),
            );

            state.log(&rec, &q_t, &w_t, &u, t);

            // eta is not constrained by the unit norm
            state.eta = state.eta + (dt * (f + f_coriolis + f_centrifugal + f_euler + f_g + u));

            // Exponential intergration to maintain unit norm of q
            state.q = state.q * dq_exp(0.5 * dt * state.eta);
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // Store viewer state on disk
        self.rerun_app.save(storage);
    }

    /// Called whenever we need repainting, which could be 60 Hz.
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // First add our panel(s):
        egui::SidePanel::right("my_side_panel")
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Simulation Parameters");
                egui::Grid::new("simulation_parameters")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Duration");
                        ui.add(egui::DragValue::new(&mut self.duration).speed(0.1));
                        ui.end_row();
                        ui.label("Timestep");
                        ui.add(egui::DragValue::new(&mut self.dt).speed(0.1));
                        ui.end_row();
                    });
                ui.heading("Initial Conditions");
                egui::Grid::new("initial_state")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.heading("Body Attitude");
                        ui.end_row();
                        ui.label("φ");
                        ui.add(egui::DragValue::new(&mut self.roll).speed(0.1));
                        ui.end_row();
                        ui.label("θ");
                        ui.add(egui::DragValue::new(&mut self.pitch).speed(0.1));
                        ui.end_row();
                        ui.label("ψ");
                        ui.add(egui::DragValue::new(&mut self.yaw).speed(0.1));
                        ui.end_row();
                        ui.heading("Body Rates");
                        ui.end_row();
                        ui.label("p");
                        ui.add(egui::DragValue::new(&mut self.omega[0]).speed(0.1));
                        ui.end_row();
                        ui.label("q");
                        ui.add(egui::DragValue::new(&mut self.omega[1]).speed(0.1));
                        ui.end_row();
                        ui.label("r");
                        ui.add(egui::DragValue::new(&mut self.omega[2]).speed(0.1));
                        ui.end_row();
                        ui.heading("Inertial Position");
                        ui.end_row();
                        ui.label("x");
                        ui.add(egui::DragValue::new(&mut self.position[0]).speed(0.1));
                        ui.end_row();
                        ui.label("y");
                        ui.add(egui::DragValue::new(&mut self.position[1]).speed(0.1));
                        ui.end_row();
                        ui.label("z");
                        ui.add(egui::DragValue::new(&mut self.position[2]).speed(0.1));
                        ui.end_row();
                        ui.heading("Inertial Velocity");
                        ui.end_row();
                        ui.label("x");
                        ui.add(egui::DragValue::new(&mut self.velocity[0]).speed(0.1));
                        ui.end_row();
                        ui.label("y");
                        ui.add(egui::DragValue::new(&mut self.velocity[1]).speed(0.1));
                        ui.end_row();
                        ui.label("z");
                        ui.add(egui::DragValue::new(&mut self.velocity[2]).speed(0.1));
                        ui.end_row();
                    });

                ui.with_layout(
                    egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                    |ui| {
                        if ui.button("Simulate").clicked() {
                            self.simulate(ctx.clone());
                        }
                    },
                )
            });

        // Now show the Rerun Viewer in the remaining space:
        self.rerun_app.update(ctx, frame);
    }
}
