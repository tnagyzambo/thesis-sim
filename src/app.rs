use na::{DualQuaternion, Matrix3, Quaternion, UnitQuaternion, Vector3};
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
            duration: 30.0,
            dt: 0.01,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            omega: Vector3::<f64>::zeros(),
            position: Vector3::<f64>::zeros(),
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
            "3d/drone",
            &rerun::Asset3D::from_file("assets/drone.stl").unwrap(),
        )
        .unwrap();
        rec.log_static(
            "3d/interial_frame",
            &rerun::Arrows3D::from_vectors(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .with_origins(&[[0.0, 0.0, 0.0]])
                .with_colors([
                    rerun::Color::from_rgb(255, 0, 0),
                    rerun::Color::from_rgb(0, 255, 0),
                    rerun::Color::from_rgb(0, 0, 255),
                ]),
        )
        .unwrap();

        let kp_r = Vector3::new(100.0, 100.0, 3.16);
        let kp_d = Vector3::new(3.1622, 3.1622, 3.1622);

        let kv_r = Vector3::new(14.142, 14.142, 2.51);
        let kv_d = Vector3::new(2.7063, 2.7063, 2.7063);

        for i in 0..n {
            state.log(&rec, i as f64 * dt);

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

            let tau_u = -J
                * (Matrix3::<f64>::from_diagonal(&kp_r)
                    * q_ln(UnitQuaternion::new_normalize(state.q.real))
                    * 2.0
                    + Matrix3::<f64>::from_diagonal(&kv_r) * state.angular_velocity_body()
                    - a);

            let f_u = -M
                * (Matrix3::<f64>::from_diagonal(&kp_d) * state.position_body()
                    + Matrix3::<f64>::from_diagonal(&kv_d)
                        * (state.angular_velocity_body().cross(&state.position_body())
                            + state.velocity_body())
                    - (a.cross(&state.position_body())
                        + state.angular_velocity_body().cross(&state.velocity_body())));

            let f_ui = state.rotation() * f_u;

            let q_t = UnitQuaternion::new_normalize(Quaternion::<f64>::from_parts(
                (1.0 / M) * (Vector3::z_axis().dot(&f_ui) + f_ui.norm()),
                (1.0 / M) * Vector3::<f64>::z_axis().cross(&f_u),
            ));

            let u = DualQuaternion::from_real_and_dual(
                Quaternion::from_imag(tau_u),
                Quaternion::from_imag(Vector3::<f64>::new(0.0, 0.0, f_ui.norm())),
                //Quaternion::from_imag(Vector3::<f64>::zeros()),
            );

            // eta is not constrained by the unit norm
            state.eta = state.eta + dt * (f + u);

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
