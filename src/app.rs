use anyhow::Result;
use na::{DualQuaternion, Matrix3, Quaternion, UnitQuaternion, Vector3};
use nalgebra as na;
use re_viewer::external::{eframe, egui};

mod plot;
mod state;

use state::{dq_exp, q_ln, State, J, J_INV, M, Q_INVERT};

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
            duration: 5.0,
            dt: 0.01,
            roll: 0.00,
            pitch: 0.0,
            yaw: 0.0,
            omega: Vector3::<f64>::new(0.0, 0.0, 0.0),
            position: Vector3::<f64>::new(0.0, 0.0, 0.0),
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

        let k_q = 10.0;
        let k_w = 3.0;
        let k_p = 1.0;
        let k_d = 2.0;
        let k_1 = 0.33;
        let k_2 = 1.0;

        let mut w_d_prev = Vector3::<f64>::zeros();
        let mut w_body_prev = self.omega;

        for i in 0..n {
            let t = i as f64 * dt;

            let wdot_body = (state.rate() - w_body_prev) / dt;
            w_body_prev = state.rate();

            let f_coriolis = -2.0 * M * state.rate().cross(&state.velocity_body());

            let f_centrifugal = -M
                * state
                    .rate()
                    .cross(&state.rate().cross(&state.position_body()));

            let f_euler = -M * wdot_body.cross(&state.position_body());

            let f_g = M * (state.attitude().conjugate() * Vector3::<f64>::new(0.0, 0.0, 9.81));

            // POS TARGETS
            let p_t = Vector3::<f64>::new(1.0, 1.0, 1.0);
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
            let e_n = Q_INVERT * (p_t - state.position());
            let edot_n = Q_INVERT * (pdot_t - state.velocity());
            let e_d = Vector3::<f64>::new(0.0, 0.0, -e_n.norm());
            let q_d = if e_d.cross(&e_n).norm() == 0.0 {
                UnitQuaternion::<f64>::identity()
            } else {
                let theta = k_1 * (k_2 * e_n.norm()).atan();
                let axis = e_d.cross(&e_n).normalize();
                UnitQuaternion::new_normalize(Quaternion::from_parts(
                    (theta / 2.0).cos(),
                    axis * (theta / 2.0).sin(),
                ))
            };
            let q_d = Q_INVERT * q_d;

            let skew = if e_n.norm() <= 0.001 {
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

            let w_d = Q_INVERT * (skew * -(q_d.conjugate() * edot_n));

            let wdot_d = (w_d - w_d_prev) / dt;
            w_d_prev = w_d;

            // ATTITUDE CONTROLLER
            let q_e = q_d.conjugate() * state.rotation();
            let w_e = state.rate() - (q_e.conjugate() * w_d);
            let wdot_e = state.rotation() * (q_d.conjugate() * wdot_d);

            let tau_u = state
                .rate()
                .cross(&(J * state.rate()))
                + J * wdot_e
                - k_q * q_e.imag()
                //- k_q * 2.0 * q_ln(q_e)
                - k_w * w_e;

            // TRANSLATIONAL CONTROLLER
            let p_e = p_t - state.position();
            let pdot_e = pdot_t - state.velocity();

            let f_thrust =
                (M * pddot_t + Vector3::new(0.0, 0.0, M * 9.81) + k_p * p_e + k_d * pdot_e)[2];

            let f_u = Vector3::new(0.0, 0.0, -f_thrust);

            // INPUT
            let torques: &[(Vector3<f64>, &str)] = &[(tau_u, "u")];
            let forces: &[(Vector3<f64>, &str)] = &[
                (f_centrifugal, "centrifugal"),
                (-f_coriolis, "coriolis"),
                //(f_euler, "euler"),
                (f_g, "g"),
                (f_u, "u"),
            ];

            plot::plot_all(
                &rec, &state, &q_d, &w_d, &p_t, &e_n, &edot_n, &wdot_body, torques, forces, t,
            )?;

            let tau: Vec<Vector3<f64>> = torques.iter().map(|(torque, _)| *torque).collect();
            let f: Vec<Vector3<f64>> = forces.iter().map(|(force, _)| *force).collect();

            // eta is not constrained by the unit norm
            state.xi = state.xi + (dt * state.compute_wrench(tau.as_slice(), f.as_slice()));

            // Exponential intergration to maintain unit norm of q
            state.q = state.q * dq_exp(0.5 * dt * state.xi);
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
                            self.simulate(ctx.clone());
                        }
                    },
                )
            });

        // Now show the Rerun Viewer in the remaining space:
        self.rerun_app.update(ctx, frame);
    }
}
