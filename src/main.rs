use na::{
    DualQuaternion, Quaternion, SMatrix, SVector, UnitDualQuaternion, UnitQuaternion, Vector3,
};
use nalgebra as na;
use re_sdk::AsComponents;
use re_viewer::external::{
    arrow2, eframe, egui, re_chunk_store, re_entity_db, re_log, re_log_types, re_memory, re_types,
};

// By using `re_memory::AccountingAllocator` Rerun can keep track of exactly how much memory it is using,
// and prune the data store when it goes above a certain limit.
// By using `mimalloc` we get faster allocations.
#[global_allocator]
static GLOBAL: re_memory::AccountingAllocator<mimalloc::MiMalloc> =
    re_memory::AccountingAllocator::new(mimalloc::MiMalloc);

// We run the gui inside of a tokio runtime
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Direct calls using the `log` crate to stderr. Control with `RUST_LOG=debug` etc.
    re_log::setup_logging();

    // Install handlers for panics and crashes that prints to stderr and send
    // them to Rerun analytics (if the `analytics` feature is on in `Cargo.toml`).
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // Listen for TCP connections from Rerun's logging SDKs.
    // There are other ways of "feeding" the viewer though - all you need is a `re_smart_channel::Receiver`.
    let rx = re_sdk_comms::serve(
        "0.0.0.0",
        re_sdk_comms::DEFAULT_SERVER_PORT,
        Default::default(),
    )?;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_app_id("rerun_extend_viewer_ui_example"),
        ..re_viewer::native::eframe_options(None)
    };

    let startup_options = re_viewer::StartupOptions::default();

    // This is used for analytics, if the `analytics` feature is on in `Cargo.toml`
    let app_env = re_viewer::AppEnvironment::Custom("My Wrapper".to_owned());

    let window_title = "My Customized Viewer";
    eframe::run_native(
        window_title,
        native_options,
        Box::new(move |cc| {
            re_viewer::customize_eframe_and_setup_renderer(cc)?;

            let mut rerun_app = re_viewer::App::new(
                re_viewer::build_info(),
                &app_env,
                startup_options,
                cc.egui_ctx.clone(),
                cc.storage,
            );
            rerun_app.add_receiver(rx);
            Ok(Box::new(App::new(rerun_app)))
        }),
    )?;

    Ok(())
}

struct App {
    rerun_app: re_viewer::App,
    w: SVector<f32, 3>,
    t: SVector<f32, 3>,
    tdot: SVector<f32, 3>,
}

impl App {
    pub fn new(rerun_app: re_viewer::App) -> Self {
        Self {
            rerun_app,
            w: SVector::<f32, 3>::zeros(),
            t: SVector::<f32, 3>::zeros(),
            tdot: SVector::<f32, 3>::zeros(),
        }
    }

    fn simulate(&self, _ctx: egui::Context) {
        let rec = re_sdk::RecordingStreamBuilder::new("rerun_example_minimal")
            .spawn()
            .unwrap();

        let dt = 0.1;

        let mut q = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
            Quaternion::identity(),
            Quaternion::from_imag(self.t),
        ));

        for i in 0..10000 {
            let t = 2.0 * q.dual * q.real.conjugate();
            let tdot = self.tdot;
            plot_3d(i as f32 * dt, &q, &tdot, &rec);

            let twist = DualQuaternion::from_real_and_dual(
                Quaternion::from_imag(self.w),
                Quaternion::from_imag(tdot + (t.imag().cross(&self.w))),
            );

            let twist = dt * 0.5 * twist;

            let exp = UnitDualQuaternion::new_normalize(DualQuaternion::from_real_and_dual(
                twist.real.exp(),
                twist.dual * twist.real.exp(),
            ));
            q = exp * q;
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
                ui.label("Omega_x");
                ui.add(egui::DragValue::new(&mut self.w[0]).speed(0.1));
                ui.label("Omega_y");
                ui.add(egui::DragValue::new(&mut self.w[1]).speed(0.1));
                ui.label("Omega_z");
                ui.add(egui::DragValue::new(&mut self.w[2]).speed(0.1));
                ui.label("t_x");
                ui.add(egui::DragValue::new(&mut self.t[0]).speed(0.1));
                ui.label("t_y");
                ui.add(egui::DragValue::new(&mut self.t[1]).speed(0.1));
                ui.label("t_z");
                ui.add(egui::DragValue::new(&mut self.t[2]).speed(0.1));
                ui.label("tdot_x");
                ui.add(egui::DragValue::new(&mut self.tdot[0]).speed(0.1));
                ui.label("tdot_y");
                ui.add(egui::DragValue::new(&mut self.tdot[1]).speed(0.1));
                ui.label("tdot_z");
                ui.add(egui::DragValue::new(&mut self.tdot[2]).speed(0.1));

                if ui.button("Simulate").clicked() {
                    self.simulate(ctx.clone());
                }
            });

        // Now show the Rerun Viewer in the remaining space:
        self.rerun_app.update(ctx, frame);
    }
}

fn plot_3d(
    time: f32,
    q: &UnitDualQuaternion<f32>,
    tdot: &SVector<f32, 3>,
    rec: &rerun::RecordingStream,
) {
    //let t = q.translation().vector; // Body frame origin expressed in inertial frame
    let t = 2.0 * q.dual * q.real.conjugate();
    let tdot = tdot;
    let tdot: [f32; 3] = std::array::from_fn(|i| tdot[i] as f32);
    let t_neg = -t; // Body frame origin expressed in inertial frame
    let t: [f32; 3] = std::array::from_fn(|i| t[i] as f32);
    let t_neg: [f32; 3] = std::array::from_fn(|i| t_neg[i] as f32);
    let x = q.rotation() * Vector3::x_axis(); // Body frame x-axis expressed in inertial frame
    let x: [f32; 3] = std::array::from_fn(|i| x[i] as f32);
    let y = q.rotation() * Vector3::y_axis(); // Body frame y-axis expressed in inertial frame
    let y: [f32; 3] = std::array::from_fn(|i| y[i] as f32);
    let z = q.rotation() * Vector3::z_axis(); // Body frame z-axis expressed in inertial frame
    let z: [f32; 3] = std::array::from_fn(|i| z[i] as f32);
    rec.set_time_seconds("sim_time", time);
    rec.log(
        "3d",
        &rerun::Arrows3D::from_vectors([
            rerun::Vector3D::from(x),
            rerun::Vector3D::from(y),
            rerun::Vector3D::from(z),
            rerun::Vector3D::from(t_neg),
            rerun::Vector3D::from(tdot),
        ])
        .with_origins([rerun::Position3D::from(t)])
        .with_colors([
            rerun::Color::from_rgb(255, 0, 0),
            rerun::Color::from_rgb(0, 255, 0),
            rerun::Color::from_rgb(0, 0, 255),
            rerun::Color::from_rgb(255, 255, 0),
            rerun::Color::from_rgb(0, 255, 255),
        ]),
    )
    .unwrap();
    let q = q.real.as_vector(); // Body frame origin expressed in inertial frame
    let q: [f32; 4] = std::array::from_fn(|i| q[i] as f32);
    rec.log(
        "box",
        &rerun::Boxes3D::from_centers_and_sizes([rerun::Vec3D::from(t)], [(1.0, 1.0, 0.1)])
            .with_colors([rerun::Color::from_rgb(255, 255, 255)])
            .with_quaternions([rerun::Quaternion::from_xyzw(q)]),
    )
    .unwrap();
}
