// SPDX-License-Identifier: Apache-2.0
//! Tie child processes to this one's lifetime.
//!
//! `llama-server` holds gigabytes of graphics memory and loopback port 8740. Dropping its
//! handle kills it on an orderly exit, but nothing runs on an *inorderly* one - "End task"
//! in Task Manager, a crash, a stop from a debugger - and the survivor then holds the GPU
//! and the port against the next launch. On a 4 GB card that is the difference between Zen
//! starting and Zen refusing to.
//!
//! A Windows job object with `KILL_ON_JOB_CLOSE` moves that guarantee into the kernel: the
//! handle is closed by process teardown however the process ends, and every assigned child
//! goes with it. The native ASR and TTS workers already notice a closed command socket and
//! exit themselves; they are assigned here too, because a worker wedged inside a DLL loader
//! never reaches the code that would notice.
#![cfg(windows)]

use std::sync::OnceLock;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Held for the life of the process on purpose: closing it is what kills the children.
struct Job(HANDLE);
// A job handle is not tied to the thread that made it.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

static JOB: OnceLock<Option<Job>> = OnceLock::new();

fn job() -> Option<&'static Job> {
    JOB.get_or_init(|| unsafe {
        let handle = CreateJobObjectW(None, windows::core::PCWSTR::null()).ok()?;
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .ok()?;
        Some(Job(handle))
    })
    .as_ref()
}

/// Assign a freshly spawned child so it cannot outlive this process.
///
/// Best effort. A machine whose policy already puts Zen in a job that forbids breakaway will
/// refuse the assignment, and that is not a reason to fail a session the user asked for -
/// the ordinary teardown path still stops the child.
pub fn adopt(handle: std::os::windows::io::RawHandle) {
    if let Some(job) = job() {
        unsafe {
            let _ = AssignProcessToJobObject(job.0, HANDLE(handle));
        }
    }
}
