use na::{DualQuaternion, Quaternion, Vector3};
use nalgebra as na;
use re_viewer::external::{eframe, egui};

mod state;

use state::{dq_exp, State};

pub struct App {
    rerun_app: re_viewer::App,
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

        let dt = 0.01;

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

        for i in 0..1000 {
            state.log(&rec, i as f64 * dt);

            let f = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(state.angular_velocity_body().cross(&state.velocity_body())),
            );

            let f_coriolis = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    -2.0 * state.angular_velocity_body().cross(&state.velocity_body()),
                ),
            );

            let f_centrifugal = DualQuaternion::from_real_and_dual(
                Quaternion::new(0.0, 0.0, 0.0, 0.0),
                Quaternion::from_imag(
                    -state
                        .angular_velocity_body()
                        .cross(&state.angular_velocity_body().cross(&state.position_body())),
                ),
            );

            // eta is not constrained by the unit norm
            state.eta = state.eta + dt * (f + f_coriolis + f_centrifugal);

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
