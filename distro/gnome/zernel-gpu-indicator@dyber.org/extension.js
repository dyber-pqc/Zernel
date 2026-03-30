/*
 * Zernel GPU Indicator — GNOME Shell Extension
 * Copyright (C) 2026 Dyber, Inc.
 *
 * Displays NVIDIA GPU utilization, temperature, and memory usage
 * in the GNOME top bar. Polls nvidia-smi every 2 seconds.
 *
 * Install: cp -r zernel-gpu-indicator@dyber.io ~/.local/share/gnome-shell/extensions/
 * Enable:  gnome-extensions enable zernel-gpu-indicator@dyber.io
 */

import GLib from 'gi://GLib';
import St from 'gi://St';
import Clutter from 'gi://Clutter';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';

const POLL_INTERVAL_MS = 2000;

// Parse nvidia-smi CSV output
function parseNvidiaSmi(output) {
    const gpus = [];
    const lines = output.trim().split('\n');
    for (const line of lines) {
        const fields = line.split(',').map(f => f.trim());
        if (fields.length >= 5) {
            gpus.push({
                index: parseInt(fields[0]) || 0,
                name: fields[1] || 'GPU',
                utilization: parseInt(fields[2]) || 0,
                memUsed: parseInt(fields[3]) || 0,
                memTotal: parseInt(fields[4]) || 0,
                temperature: parseInt(fields[5]) || 0,
            });
        }
    }
    return gpus;
}

export default class ZernelGpuIndicator extends Extension {
    enable() {
        this._indicator = new PanelMenu.Button(0.0, 'Zernel GPU', false);

        // Top bar label
        this._label = new St.Label({
            text: 'GPU: --',
            y_align: Clutter.ActorAlign.CENTER,
            style_class: 'zernel-gpu-label',
        });
        this._indicator.add_child(this._label);

        // Dropdown menu
        this._menuTitle = new PopupMenu.PopupMenuItem('Zernel GPU Monitor', { reactive: false });
        this._menuTitle.label.style_class = 'zernel-gpu-popup-title';
        this._indicator.menu.addMenuItem(this._menuTitle);
        this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._gpuItems = [];

        // Add to panel
        Main.panel.addToStatusArea('zernel-gpu-indicator', this._indicator);

        // Start polling
        this._pollGpu();
        this._timer = GLib.timeout_add(GLib.PRIORITY_DEFAULT, POLL_INTERVAL_MS, () => {
            this._pollGpu();
            return GLib.SOURCE_CONTINUE;
        });
    }

    disable() {
        if (this._timer) {
            GLib.source_remove(this._timer);
            this._timer = null;
        }
        if (this._indicator) {
            this._indicator.destroy();
            this._indicator = null;
        }
    }

    _pollGpu() {
        try {
            const [ok, stdout] = GLib.spawn_command_line_sync(
                'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits'
            );

            if (!ok) {
                this._label.set_text('GPU: N/A');
                return;
            }

            const output = new TextDecoder().decode(stdout);
            const gpus = parseNvidiaSmi(output);

            if (gpus.length === 0) {
                this._label.set_text('GPU: N/A');
                return;
            }

            // Update top bar — show first GPU summary
            const g0 = gpus[0];
            const memPct = g0.memTotal > 0 ? Math.round(g0.memUsed / g0.memTotal * 100) : 0;
            this._label.set_text(`GPU: ${g0.utilization}% ${g0.temperature}°C ${memPct}%mem`);

            // Color coding
            if (g0.utilization > 90 || g0.temperature > 80) {
                this._label.style_class = 'zernel-gpu-label-crit';
            } else if (g0.utilization > 70 || g0.temperature > 70) {
                this._label.style_class = 'zernel-gpu-label-warn';
            } else {
                this._label.style_class = 'zernel-gpu-label';
            }

            // Update dropdown menu
            // Remove old GPU items
            for (const item of this._gpuItems) {
                item.destroy();
            }
            this._gpuItems = [];

            for (const gpu of gpus) {
                const memGb = (gpu.memUsed / 1024).toFixed(1);
                const totalGb = (gpu.memTotal / 1024).toFixed(1);
                const text = `GPU ${gpu.index}: ${gpu.name}  |  ${gpu.utilization}%  ${gpu.temperature}°C  ${memGb}/${totalGb} GB`;
                const item = new PopupMenu.PopupMenuItem(text, { reactive: false });
                item.label.style_class = 'zernel-gpu-popup-row';
                this._indicator.menu.addMenuItem(item);
                this._gpuItems.push(item);
            }

            // Separator + links
            if (this._gpuItems.length > 0 && !this._footerAdded) {
                this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
                const dashItem = new PopupMenu.PopupMenuItem('Open Zernel Dashboard');
                dashItem.connect('activate', () => {
                    GLib.spawn_command_line_async('xdg-open http://localhost:3000');
                });
                this._indicator.menu.addMenuItem(dashItem);

                const watchItem = new PopupMenu.PopupMenuItem('Open zernel watch (Terminal)');
                watchItem.connect('activate', () => {
                    GLib.spawn_command_line_async('gnome-terminal -- zernel watch');
                });
                this._indicator.menu.addMenuItem(watchItem);

                this._footerAdded = true;
            }

        } catch (e) {
            this._label.set_text('GPU: err');
        }
    }
}
