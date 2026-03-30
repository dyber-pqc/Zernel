// Copyright (C) 2026 Dyber, Inc. — Proprietary

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

/// Simulated GPU data for demo mode (when zerneld is not running).
struct DemoState {
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

struct GpuInfo {
    id: u32,
    util: u8,
    mem_used_gb: f64,
    mem_total_gb: f64,
}

impl DemoState {
    fn new() -> Self {
        Self {
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

    fn tick(&mut self) {
        self.tick += 1;
        self.step = (self.step + 3).min(self.total_steps);
        self.loss = (1.8 * (-0.0002 * self.step as f64).exp()).max(0.3);

        // Fluctuate GPU utilization
        for gpu in &mut self.gpus {
            let jitter = ((self.tick * (gpu.id as u64 + 1) * 7) % 6) as i8 - 3;
            gpu.util = (gpu.util as i8 + jitter).clamp(85, 99) as u8;
        }

        // Cycle phases
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
}

/// Launch the full-screen Ratatui dashboard.
pub async fn run() -> Result<()> {
    info!("starting watch dashboard");

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = DemoState::new();

    // Main render loop
    loop {
        state.tick();

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // title
                    Constraint::Length(6), // GPU bars
                    Constraint::Length(5), // training metrics
                    Constraint::Length(6), // eBPF telemetry
                    Constraint::Min(3),    // scheduler
                ])
                .split(f.area());

            // Title
            let elapsed_mins = state.tick / 6; // ~10 ticks/sec
            let title = Paragraph::new(Line::from(vec![
                Span::styled(
                    " Zernel Watch ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "  job: demo-training  |  step: {}/{}  |  elapsed: {}m",
                    state.step, state.total_steps, elapsed_mins
                )),
            ]))
            .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            // GPU utilization
            let gpu_constraints: Vec<Constraint> = state
                .gpus
                .iter()
                .map(|_| Constraint::Percentage(25))
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

            // Training metrics
            let loss_str = format!("{:.4}", state.loss);
            let progress = state.step as f64 / state.total_steps as f64;
            let eta_steps = state.total_steps - state.step;
            let eta_mins = eta_steps / 3 / 6; // rough estimate

            let metrics_text = vec![
                Line::from(vec![
                    Span::styled(" loss: ", Style::default().fg(Color::Yellow)),
                    Span::raw(&loss_str),
                    Span::raw(format!("   step: {}/{}", state.step, state.total_steps)),
                    Span::raw(format!("   ETA: {eta_mins}m")),
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
        })?;

        // Handle input (non-blocking, 100ms timeout)
        if event::poll(std::time::Duration::from_millis(166))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') => state = DemoState::new(),
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
