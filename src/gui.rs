use eframe::egui;
use na::{
    DualQuaternion, Quaternion, SMatrix, SVector, UnitDualQuaternion, UnitQuaternion, Vector3,
};
use nalgebra as na;

pub struct App {
    w: SVector<f32, 3>,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Rerun recorder

        Self {
            w: SVector::<f32, 3>::zeros(),
        }
    }

    fn simulate(&self, _ctx: egui::Context) {
        let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal")
            .spawn()
            .unwrap();

        let dt = 0.1;
        let mut q = UnitDualQuaternion::identity();

        let t_b = SVector::<f32, 3>::zeros();
        let tdot_b = SVector::<f32, 3>::new(1.0, 0.0, 0.0);

        for _ in 0..100 {
            let x_axis = Vector3::x_axis();
            let y_axis = Vector3::y_axis();
            let z_axis = Vector3::z_axis();

            let p = q.dual.vector();
            let p_: [f32; 3] = std::array::from_fn(|i| p[i] as f32);
            let x_ = q.transform_vector(&x_axis);
            let x_: [f32; 3] = std::array::from_fn(|i| x_[i] as f32);
            let y_ = q * y_axis;
            let y_: [f32; 3] = std::array::from_fn(|i| y_[i] as f32);
            let z_ = q * z_axis;
            let z_: [f32; 3] = std::array::from_fn(|i| z_[i] as f32);
            rec.log(
                "quat",
                &rerun::Arrows3D::from_vectors([
                    rerun::Vector3D::from(x_),
                    rerun::Vector3D::from(y_),
                    rerun::Vector3D::from(z_),
                ])
                .with_origins([rerun::Position3D::from(p_)])
                .with_colors([
                    rerun::Color::from_rgb(255, 0, 0),
                    rerun::Color::from_rgb(0, 255, 0),
                    rerun::Color::from_rgb(0, 0, 255),
                ]),
            )
            .unwrap();

            let omega = DualQuaternion::from_real_and_dual(
                0.5 * Quaternion::from_parts(0.0, self.w),
                0.5 * Quaternion::from_parts(0.0, self.w.cross(&t_b) + tdot_b),
            );

            q = UnitDualQuaternion::new_normalize(q.dual_quaternion() + omega * dt);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("data")
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Omega_x");
                ui.add(
                    egui::DragValue::new(&mut self.w[0])
                        .speed(0.1)
                        .range(-1.0..=1.0),
                );
                ui.label("Omega_y");
                ui.add(
                    egui::DragValue::new(&mut self.w[1])
                        .speed(0.1)
                        .range(-1.0..=1.0),
                );
                ui.label("Omega_z");
                ui.add(
                    egui::DragValue::new(&mut self.w[2])
                        .speed(0.1)
                        .range(-1.0..=1.0),
                );

                if ui.button("Simulate").clicked() {
                    self.simulate(ctx.clone());
                }
            });

        egui::CentralPanel::default().show(ctx, |_| {});
    }
}
