// Copyright (C) 2026 Dyber, Inc. — Proprietary

use crate::telemetry::client::{self, TelemetryClient, TelemetrySnapshot};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Terminal,
};
use std::io;
use tracing::info;

/// Dashboard state — either live from zerneld or simulated demo data.
struct DashboardState {
    mode: DashboardMode,
    tick: u64,
    gpus: Vec<GpuInfo>,
    loss: f64,
    step: u64,
    total_steps: u64,
    cuda_p50_us: f64,
    cuda_p99_us: f64,
    nccl_p50_ms: f64,
    nccl_p99_ms: f64,
    dataloader_ms: f64,
    pcie_gbps: f64,
    phase: String,
}

enum DashboardMode {
    Demo,
    Live,
}

struct GpuInfo {
    id: u32,
    util: u8,
    mem_used_gb: f64,
    mem_total_gb: f64,
}

impl DashboardState {
    fn new_demo() -> Self {
        Self {
            mode: DashboardMode::Demo,
            tick: 0,
            gpus: vec![
                GpuInfo {
                    id: 0,
                    util: 94,
                    mem_used_gb: 78.2,
                    mem_total_gb: 80.0,
                },
                GpuInfo {
                    id: 1,
                    util: 91,
                    mem_used_gb: 77.8,
                    mem_total_gb: 80.0,
                },
                GpuInfo {
                    id: 2,
                    util: 96,
                    mem_used_gb: 79.1,
                    mem_total_gb: 80.0,
                },
                GpuInfo {
                    id: 3,
                    util: 93,
                    mem_used_gb: 78.9,
                    mem_total_gb: 80.0,
                },
            ],
            loss: 1.8,
            step: 0,
            total_steps: 10000,
            cuda_p50_us: 142.0,
            cuda_p99_us: 891.0,
            nccl_p50_ms: 34.0,
            nccl_p99_ms: 67.0,
            dataloader_ms: 8.0,
            pcie_gbps: 31.2,
            phase: "GpuCompute".into(),
        }
    }

    fn new_live() -> Self {
        Self {
            mode: DashboardMode::Live,
            tick: 0,
            gpus: Vec::new(),
            loss: 0.0,
            step: 0,
            total_steps: 0,
            cuda_p50_us: 0.0,
            cuda_p99_us: 0.0,
            nccl_p50_ms: 0.0,
            nccl_p99_ms: 0.0,
            dataloader_ms: 0.0,
            pcie_gbps: 0.0,
            phase: "Unknown".into(),
        }
    }

    /// Update from a real zerneld telemetry snapshot.
    fn apply_snapshot(&mut self, snap: &TelemetrySnapshot) {
        self.tick += 1;
        self.cuda_p50_us = snap.cuda_latency_p50_us;
        self.cuda_p99_us = snap.cuda_latency_p99_us;
        self.nccl_p50_ms = snap.nccl_allreduce_p50_ms;
        self.nccl_p99_ms = snap.nccl_allreduce_p99_ms;
        self.dataloader_ms = snap.dataloader_wait_p50_ms;

        // Convert GPU memory entries to display info
        self.gpus.clear();
        for (i, entry) in snap.gpu_utilization.iter().enumerate() {
            let total_gb = if entry.peak_bytes > 0 {
                entry.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            } else {
                80.0
            };
            let used_gb = entry.current_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let util = if entry.peak_bytes > 0 {
                ((entry.current_bytes as f64 / entry.peak_bytes as f64) * 100.0) as u8
            } else {
                0
            };
            self.gpus.push(GpuInfo {
                id: i as u32,
                util,
                mem_used_gb: used_gb,
                mem_total_gb: total_gb,
            });
        }
    }

    /// Advance demo state by one tick.
    fn demo_tick(&mut self) {
        self.tick += 1;
        self.step = (self.step + 3).min(self.total_steps);
        self.loss = (1.8 * (-0.0002 * self.step as f64).exp()).max(0.3);

        for gpu in &mut self.gpus {
            let jitter = ((self.tick * (gpu.id as u64 + 1) * 7) % 6) as i8 - 3;
            gpu.util = (gpu.util as i8 + jitter).clamp(85, 99) as u8;
        }

        let phase_cycle = self.tick % 20;
        self.phase = match phase_cycle {
            0..=2 => "DataLoading",
            3..=14 => "GpuCompute",
            15..=16 => "NcclCollective",
            17..=18 => "OptimizerStep",
            _ => "GpuCompute",
        }
        .into();
    }

    fn mode_label(&self) -> &str {
        match self.mode {
            DashboardMode::Demo => " [DEMO] ",
            DashboardMode::Live => " [LIVE] ",
        }
    }
}

/// Launch the full-screen Ratatui dashboard.
pub async fn run() -> Result<()> {
    info!("starting watch dashboard");

    // Try to connect to zerneld
    let port = client::ws_port();
    let client = TelemetryClient::new("127.0.0.1", port);
    let (mut state, mut rx) = match client.try_connect().await {
        Some(rx) => {
            info!(port, "connected to zerneld");
            (DashboardState::new_live(), Some(rx))
        }
        None => {
            info!("zerneld not available, using demo mode");
            (DashboardState::new_demo(), None)
        }
    };

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Main render loop
    loop {
        // Update state
        match &mut rx {
            Some(receiver) => {
                // Try to drain latest snapshot (non-blocking)
                while let Ok(snap) = receiver.try_recv() {
                    state.apply_snapshot(&snap);
                }
            }
            None => {
                state.demo_tick();
            }
        }

        terminal.draw(|f| render_dashboard(f, &state))?;

        // Handle input (non-blocking, 166ms = ~6fps)
        if event::poll(std::time::Duration::from_millis(166))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') => {
                            state = match state.mode {
                                DashboardMode::Demo => DashboardState::new_demo(),
                                DashboardMode::Live => DashboardState::new_live(),
                            };
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

fn render_dashboard(f: &mut ratatui::Frame, state: &DashboardState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Min(3),
        ])
        .split(f.area());

    // Title
    let elapsed_mins = state.tick / 6;
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " Zernel Watch ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            state.mode_label(),
            Style::default().fg(match state.mode {
                DashboardMode::Demo => Color::Yellow,
                DashboardMode::Live => Color::Green,
            }),
        ),
        Span::raw(format!(
            " step: {}/{}  |  elapsed: {}m",
            state.step, state.total_steps, elapsed_mins
        )),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // GPU utilization
    if !state.gpus.is_empty() {
        let gpu_pct = 100 / state.gpus.len().max(1) as u16;
        let gpu_constraints: Vec<Constraint> = state
            .gpus
            .iter()
            .map(|_| Constraint::Percentage(gpu_pct))
            .collect();
        let gpu_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(gpu_constraints)
            .split(chunks[1]);

        for (i, gpu) in state.gpus.iter().enumerate() {
            let color = if gpu.util > 90 {
                Color::Green
            } else if gpu.util > 70 {
                Color::Yellow
            } else {
                Color::Red
            };
            let gauge = Gauge::default()
                .block(
                    Block::default()
                        .title(format!(
                            "GPU {} | {:.1}/{:.1} GB",
                            gpu.id, gpu.mem_used_gb, gpu.mem_total_gb
                        ))
                        .borders(Borders::ALL),
                )
                .gauge_style(Style::default().fg(color))
                .percent(gpu.util as u16)
                .label(format!("{}%", gpu.util));
            f.render_widget(gauge, gpu_chunks[i]);
        }
    } else {
        let msg = Paragraph::new(" Waiting for GPU data...")
            .block(Block::default().title(" GPUs ").borders(Borders::ALL));
        f.render_widget(msg, chunks[1]);
    }

    // Training metrics
    let loss_str = format!("{:.4}", state.loss);
    let progress = if state.total_steps > 0 {
        state.step as f64 / state.total_steps as f64
    } else {
        0.0
    };

    let metrics_text = vec![
        Line::from(vec![
            Span::styled(" loss: ", Style::default().fg(Color::Yellow)),
            Span::raw(&loss_str),
            Span::raw(format!("   step: {}/{}", state.step, state.total_steps)),
        ]),
        Line::from(vec![
            Span::styled(" progress: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:.1}%", progress * 100.0)),
        ]),
    ];
    let metrics = Paragraph::new(metrics_text).block(
        Block::default()
            .title(" Training Metrics ")
            .borders(Borders::ALL),
    );
    f.render_widget(metrics, chunks[2]);

    // eBPF telemetry
    let telem_text = vec![
        Line::from(format!(
            " CUDA launch: p50={:.0}us  p99={:.0}us    DataLoader wait: p50={:.0}ms",
            state.cuda_p50_us, state.cuda_p99_us, state.dataloader_ms
        )),
        Line::from(format!(
            " NCCL allreduce: p50={:.0}ms  p99={:.0}ms    PCIe BW: {:.1} GB/s",
            state.nccl_p50_ms, state.nccl_p99_ms, state.pcie_gbps
        )),
    ];
    let telem = Paragraph::new(telem_text).block(
        Block::default()
            .title(" eBPF Telemetry ")
            .borders(Borders::ALL),
    );
    f.render_widget(telem, chunks[3]);

    // Scheduler phase
    let phase_color = match state.phase.as_str() {
        "GpuCompute" => Color::Green,
        "DataLoading" => Color::Yellow,
        "NcclCollective" => Color::Cyan,
        "OptimizerStep" => Color::Magenta,
        _ => Color::White,
    };
    let sched_text = vec![Line::from(vec![
        Span::raw(" Phase: "),
        Span::styled(
            &state.phase,
            Style::default()
                .fg(phase_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("   [q] quit  [r] reset"),
    ])];
    let sched = Paragraph::new(sched_text)
        .block(Block::default().title(" Scheduler ").borders(Borders::ALL));
    f.render_widget(sched, chunks[4]);
}
