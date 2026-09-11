// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! Generates Tauri's context (window configuration, capabilities and the embedded interface
//! assets from `tauri.conf.json`) so it can be compiled into the executable.
fn main() {
    tauri_build::build()
}
