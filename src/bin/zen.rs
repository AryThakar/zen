// SPDX-License-Identifier: Apache-2.0
//! Zen. The window's event loop owns the main thread, so this is deliberately not
//! `#[tokio::main]`; the async runtime is built inside and shared with Tauri.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> std::process::ExitCode {
    zen::runtime::main_entry()
}
