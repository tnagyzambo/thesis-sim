use anyhow::Result;
use na::{Vector3, Vector4};
use nalgebra as na;
use re_viewer::external::{eframe, egui, re_log};

pub mod control;
pub mod dynamics;
pub mod measurement;
pub mod plot;
pub mod state;
pub mod ukf;

use state::{dq_exp, q_ln, Force, State, Torque, A, J, M, MOTOR_A, MOTOR_B, M_INV, Q_INVERT};

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
            duration: 2.0,
            dt: 0.01,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            omega: Vector3::<f64>::new(0.0, 0.0, 0.0),
            position: Vector3::<f64>::new(1.2, 1.4, 2.0),
            velocity: Vector3::<f64>::zeros(),
        }
    }

    fn simulate(&self, _ctx: egui::Context) -> Result<()> {
        let rec = re_sdk::RecordingStreamBuilder::new("Simulator")
            .spawn()
            .unwrap();

        plot::plot_all_static(&rec)?;

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

        let mut ukf_state = ukf::UkfState::new(*state.q.dual_quaternion());

        let mut controller_state = control::ControllerState::default();
        let mut motor_state = Vector4::<f64>::new(1500.0, 1500.0, 1500.0, 1500.0);
        let mut accl = Vector3::zeros();
        let mut rate_bias = Vector3::zeros();

        for i in 0..n {
            let t = i as f64 * dt;
            plot::plot_state(&rec, &state, &accl, t)?;

            // MEASURE
            let (noisy_accl, noisy_rate) = measurement::measurment(&state, accl, &mut rate_bias);
            let noisy_pos = if (i % 20) == 0 {
                Some(measurement::gps(&state))
            } else {
                None
            };
            plot::plot_measurments(&rec, &noisy_pos, &noisy_accl, &noisy_rate, t)?;

            // FILTER
            let filtered_state = ukf::ukf(
                &rec,
                &noisy_pos,
                &noisy_accl,
                &noisy_rate,
                &state,
                &mut ukf_state,
                dt,
                t,
            )?;

            // CONTROL
            let (f_u, tau_u) =
                control::control(&rec, &state, &mut controller_state, &mut motor_state, dt, t)?;

            // INPUT
            let torques = vec![tau_u];
            let mut forces = dynamics::ficticous_forces(&state);
            forces.push(dynamics::gravity(&state));

            if (1.2 >= t) && (t > 1.0) {
                let disturbance = Force::new(Vector3::new(10.0, 10.0, 10.0), "dist".to_string());
                forces.push(disturbance);
            }

            accl = M_INV
                * forces
                    .iter()
                    .map(|force| force.f)
                    .collect::<Vec<Vector3<f64>>>()
                    .as_slice()
                    .iter()
                    .sum::<Vector3<f64>>();
            forces.push(f_u);
            plot::plot_forces(&rec, &forces, &torques, t)?;

            // DYNAMICS
            state = dynamics::dynamics(&state, &forces, &torques, dt);
        }
        Ok(())
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
                            match self.simulate(ctx.clone()) {
                                Ok(()) => (),
                                Err(e) => re_log::error!("{}", e),
                            };
                        }
                    },
                )
            });

        // Now show the Rerun Viewer in the remaining space:
        self.rerun_app.update(ctx, frame);
    }
}
