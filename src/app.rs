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
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            omega: Vector3::<f64>::zeros(),
            position: Vector3::<f64>::new(1.0, 1.0, 1.0),
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

        let kp_r = Vector3::new(100.0, 100.0, 3.16);
        let kp_d = Vector3::new(1.1622, 1.1622, 1.1622);

        let kv_r = Vector3::new(50.142, 50.142, 2.51);
        let kv_d = Vector3::new(2.7063, 2.7063, 2.7063);

        let mut q_t_prev = UnitQuaternion::<f64>::identity();

        for i in 0..n {
            let a = -J.try_inverse().unwrap()
                * (state
                    .angular_velocity_body()
                    .cross(&(J * state.angular_velocity_body())));

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

            let f_g = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    M * (state.rotation().conjugate() * Vector3::<f64>::new(0.0, 0.0, -9.81)),
                ),
            );

            let p_e = -state.position_body();
            let v_e = -state.velocity_body();
            let f_u = M
                * (Matrix3::<f64>::from_diagonal(&kp_d) * p_e
                    + Matrix3::<f64>::from_diagonal(&kv_d)
                        * (state.angular_velocity_body().cross(&p_e) + v_e)
                    - a.cross(&state.velocity_body())
                    - state.angular_velocity_body().cross(&state.position_body()))
                - M * (state.rotation().conjugate() * Vector3::<f64>::new(0.0, 0.0, -9.81));

            let f_ui = state.rotation() * f_u;

            let q_t_scalar = (1.0 / M) * (Vector3::z_axis().dot(&f_ui) + f_ui.norm());
            let q_t_vec = (1.0 / M) * Vector3::<f64>::z_axis().cross(&f_ui);
            let q_t = if (q_t_scalar == 0.0) && (q_t_vec.norm() == 0.0) {
                UnitQuaternion::identity()
            } else {
                UnitQuaternion::new_normalize(Quaternion::<f64>::from_parts(q_t_scalar, q_t_vec))
            };

            let w_t = 2.0 * (q_ln(q_t) - q_ln(q_t_prev)) / dt;
            q_t_prev = q_t;

            let q_e = q_t * state.rotation().conjugate();
            let w_e = w_t - state.angular_velocity_body();

            let q_u = 2.0 * q_ln(q_e);
            let w_u = w_e;
            let tau_u = J
                * (Matrix3::<f64>::from_diagonal(&kp_r) * q_u
                    + Matrix3::<f64>::from_diagonal(&kv_r) * w_u
                    - (-J.try_inverse().unwrap()
                        * (state
                            .angular_velocity_body()
                            .cross(&(J * state.angular_velocity_body())))));

            let u = DualQuaternion::from_real_and_dual(
                Quaternion::from_imag(Vector3::<f64>::new(tau_u[0], tau_u[1], tau_u[2])),
                Quaternion::from_imag(Vector3::<f64>::new(0.0, 0.0, f_u.norm())),
                //Quaternion::from_imag(Vector3::<f64>::zeros()),
            );

            state.log(&rec, &q_t, &w_t, i as f64 * dt);

            // eta is not constrained by the unit norm
            state.eta = state.eta + dt * (f + f_coriolis + f_centrifugal + f_g + u);

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
