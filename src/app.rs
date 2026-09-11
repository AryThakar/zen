// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//! The desktop application: one window, one embedded webview, one engine, one process.
//!
//! The interface is compiled into this binary and loaded from a private app origin, so it
//! keeps the WebView2 capture stack without exposing a listener or network surface. Closing
//! the window leaves the engine resident behind a tray icon; Windows
//! toasts report anything the user would otherwise have missed on screen.
use crate::{
    bridge::{EngineOptions, Input, Transport},
    remote,
};
use futures_util::FutureExt;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tauri::{
    ipc::{Channel, InvokeResponseBody, Request},
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, RunEvent, WindowEvent,
};
use tauri_plugin_notification::NotificationExt;
use tokio::sync::{mpsc, watch};

/// Reply audio is produced far faster than it is played, so the queue only has to absorb
/// scheduling jitter, not a whole utterance.
const INPUT_QUEUE: usize = 256;

struct Live {
    transport: Transport,
    input: mpsc::Sender<(u64, Input)>,
    finished: watch::Receiver<bool>,
}

impl Live {
    async fn wait_finished(&self) {
        let mut finished = self.finished.clone();
        while !*finished.borrow_and_update() {
            if finished.changed().await.is_err() {
                break;
            }
        }
    }
}

type SessionInput = mpsc::Receiver<(u64, Input)>;
type Claim = (Arc<Live>, u64, Option<(SessionInput, watch::Sender<bool>)>);

#[derive(Default)]
struct SessionRegistry(Mutex<Option<Arc<Live>>>);

impl SessionRegistry {
    fn current(&self) -> Option<Arc<Live>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    async fn claim(&self, frames: Channel<InvokeResponseBody>) -> Claim {
        loop {
            let closing = {
                let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(live) = slot.as_ref() {
                    if !live.transport.status().2 {
                        return (live.clone(), live.transport.attach(frames), None);
                    }
                }
                if let Some(live) = slot.as_ref().filter(|live| {
                    live.transport.status().2
                        && !*live.finished.borrow()
                        && live.finished.has_changed().is_ok()
                }) {
                    Some(live.clone())
                } else {
                    let transport = Transport::new();
                    let revision = transport.attach(frames);
                    let (tx, rx) = mpsc::channel(INPUT_QUEUE);
                    let (finished, completion) = watch::channel(false);
                    let live = Arc::new(Live {
                        transport,
                        input: tx,
                        finished: completion,
                    });
                    *slot = Some(live.clone());
                    return (live, revision, Some((rx, finished)));
                }
            };
            if let Some(live) = closing {
                live.wait_finished().await;
            }
        }
    }

    async fn close(&self) {
        let live = {
            let slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(live) = slot.as_ref() {
                live.transport.revoke();
            }
            slot.clone()
        };
        if let Some(live) = live {
            live.wait_finished().await;
        }
    }

    fn finish(&self, live: &Live, finished: watch::Sender<bool>, failure: Option<&str>) {
        let _slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(reason) = failure.filter(|_| !live.transport.status().2) {
            // "Something went wrong" is not something anyone can act on. A missing model
            // names the file; a busy GPU says so. Pass the reason through.
            live.transport.send(
                serde_json::json!({"type":"error","code":"backend_unavailable","detail":reason}),
            );
        }
        live.transport.revoke();
        finished.send_replace(true);
    }
}

pub struct AppState {
    options: EngineOptions,
    sessions: SessionRegistry,
    exiting: AtomicBool,
    /// Set once the user has been told that closing the window did not stop Zen.
    background_notice: Mutex<bool>,
}

/// Bind the webview to a session, starting one if none is running. The returned revision
/// stamps every later frame so a reloaded webview cannot have its old audio credited.
#[tauri::command]
async fn zen_attach(
    app: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    frames: Channel<InvokeResponseBody>,
    system_prompt: Option<String>,
) -> Result<u64, String> {
    // The session may start with instructions of the user's own, but `options` keeps Zen's
    // configured default: that is what "Clear history & start fresh" with an empty box
    // restores it to, and overwriting it here leaves the persona no way back.
    let options = state.options.clone();
    let session_prompt = match system_prompt.filter(|p| !p.trim().is_empty()) {
        Some(prompt) => {
            crate::bridge::validate_prompt(&prompt)?;
            prompt
        }
        None => options.system_prompt.clone(),
    };

    if state.exiting.load(Ordering::Acquire) {
        return Err("Zen is shutting down".into());
    }
    let (live, revision, worker) = state.sessions.claim(frames).await;
    let Some((rx, finished)) = worker else {
        return Ok(revision);
    };

    let handle = app.clone();
    let state = Arc::clone(&state);
    tauri::async_runtime::spawn(async move {
        let result = std::panic::AssertUnwindSafe(remote::run(
            &options,
            &session_prompt,
            live.transport.clone(),
            rx,
        ))
        .catch_unwind()
        .await;
        let failure = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error.to_string()),
            Err(_) => Some("the engine panicked".to_string()),
        };
        state.sessions.finish(&live, finished, failure.as_deref());
        if let Some(reason) = failure {
            let _ = handle
                .notification()
                .builder()
                .title("Zen stopped")
                .body(format!("The local engine could not continue: {reason}"))
                .show();
        }
    });
    Ok(revision)
}

/// Everything the page sends arrives through this one command, as a raw body:
///
/// ```text
/// [revision u64 LE][kind u8][payload]
///   kind 0: PCM, 20 ms of mono signed 16-bit LE at 16 kHz
///   kind 1: a UTF-8 JSON control message
/// ```
///
/// One command, and the page keeps one call in flight, because order is load-bearing.
/// IPC calls are independent promises that can be delivered in any order, and the turn
/// logic depends on capture, `end_audio` and acknowledgements arriving exactly as they
/// were produced: an `end_audio` that overtakes the frames it follows truncates the
/// user's last word, and an acknowledgement that arrives early fails the turn. A socket
/// gave this ordering for free; IPC has to be asked for it.
#[tauri::command]
async fn zen_input(
    state: tauri::State<'_, Arc<AppState>>,
    request: Request<'_>,
) -> Result<(), String> {
    let tauri::ipc::InvokeBody::Raw(frame) = request.body() else {
        return Err("input must be raw bytes".into());
    };
    let (revision, input) = crate::bridge::parse_input_frame(frame)?;
    let Some(live) = state.sessions.current() else {
        return Err("no session".into());
    };
    if !live.transport.accepts(revision) {
        return Ok(());
    }
    // Backpressure the one in-flight command instead of silently losing controls.
    tokio::time::timeout(Duration::from_secs(2), live.input.send((revision, input)))
        .await
        .map_err(|_| "engine input stalled".to_string())?
        .map_err(|_| "session ended".to_string())
}

/// Release this webview without ending the session, so a reload cannot have its old
/// frames credited to the new page.
#[tauri::command]
fn zen_detach(state: tauri::State<'_, Arc<AppState>>, revision: u64) {
    if let Some(live) = state.sessions.current() {
        live.transport.detach(revision);
    }
}

/// End the session and everything holding conversation state. The next attach starts a
/// fresh engine, so history, KV cache and native speech state do not carry over.
#[tauri::command]
async fn zen_disconnect(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.sessions.close().await;
    Ok(())
}

/// WebView2 asks the host before granting a page the microphone. With no handler attached
/// the request is never answered and `getUserMedia` hangs forever instead of failing, so
/// the embedded UI has to answer for itself. There is no untrusted page here: the only
/// document loaded is the one compiled into this binary.
#[cfg(windows)]
fn grant_microphone(window: &tauri::WebviewWindow) -> Result<(), String> {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_PERMISSION_KIND_MICROPHONE, COREWEBVIEW2_PERMISSION_STATE_ALLOW,
        COREWEBVIEW2_PERMISSION_STATE_DENY,
    };
    use webview2_com::PermissionRequestedEventHandler;

    window
        .with_webview(move |webview| {
            let handler = PermissionRequestedEventHandler::create(Box::new(|_, args| unsafe {
                let Some(args) = args else { return Ok(()) };
                let mut kind = Default::default();
                args.PermissionKind(&mut kind)?;
                if kind == COREWEBVIEW2_PERMISSION_KIND_MICROPHONE {
                    let mut uri = windows::core::PWSTR::null();
                    args.Uri(&mut uri)?;
                    let trusted = tauri::Url::parse(&webview2_com::take_pwstr(uri))
                        .is_ok_and(|url| trusted_origin(&url));
                    args.SetState(if trusted {
                        COREWEBVIEW2_PERMISSION_STATE_ALLOW
                    } else {
                        COREWEBVIEW2_PERMISSION_STATE_DENY
                    })?;
                }
                Ok(())
            }));
            let outcome = (|| unsafe {
                let core = webview.controller().CoreWebView2()?;
                let mut token = Default::default();
                core.add_PermissionRequested(&handler, &mut token)?;
                Ok::<_, windows::core::Error>(())
            })();
            if outcome.is_err() {
                eprintln!("Microphone permission handler could not be installed");
            }
        })
        .map_err(|e| e.to_string())
}

fn trusted_origin(url: &tauri::Url) -> bool {
    matches!(
        (url.scheme(), url.host_str()),
        ("http" | "https", Some("tauri.localhost")) | ("tauri", Some("localhost"))
    ) && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
}

fn show_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

pub fn run(options: EngineOptions) -> Result<(), crate::bridge::Error> {
    options.validate()?;
    let run_for = options.run_for;
    let state = Arc::new(AppState {
        options,
        sessions: SessionRegistry::default(),
        exiting: AtomicBool::new(false),
        background_notice: Mutex::new(false),
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // One engine owns the models and the GPU. A second launch raises the first.
            show_window(app);
        }))
        .manage(Arc::clone(&state))
        .invoke_handler(tauri::generate_handler![
            zen_attach,
            zen_input,
            zen_detach,
            zen_disconnect
        ])
        .setup(move |app| {
            // Tauri creates the configured main window before setup. Reusing it keeps
            // WebView2 initialization on the framework's normal path (manual creation
            // here can fail with Access Denied on Windows).
            let window = app
                .get_webview_window("main")
                .ok_or("main window is missing")?;
            #[cfg(windows)]
            if let Err(reason) = grant_microphone(&window) {
                // Without this the microphone silently never opens, so say so plainly
                // rather than letting the UI wait on a promise that cannot settle.
                eprintln!("Microphone permission handler not installed: {reason}");
            }

            let show = MenuItem::with_id(app, "show", "Open Zen", true, None::<&str>)
                .map_err(|error| format!("could not create tray show item: {error}"))?;
            let quit = MenuItem::with_id(app, "quit", "Quit Zen", true, None::<&str>)
                .map_err(|error| format!("could not create tray quit item: {error}"))?;
            let menu = Menu::with_items(app, &[&show, &quit])
                .map_err(|error| format!("could not create tray menu: {error}"))?;
            TrayIconBuilder::new()
                .icon(
                    app.default_window_icon()
                        .ok_or("missing icon")
                        .map_err(|error| error.to_string())?
                        .clone(),
                )
                .tooltip("Zen")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        button_state: tauri::tray::MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_window(tray.app_handle());
                    }
                })
                .build(app)
                .map_err(|error| format!("could not create tray icon: {error}"))?;

            // A bounded run has to end the process, not just the session: the window
            // and the tray icon outlive any one conversation.
            if let Some(limit) = run_for {
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(limit).await;
                    handle.exit(0);
                });
            }

            let _ = window;
            Ok(())
        })
        .on_window_event(move |window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                // Closing the window must not tear down a conversation in progress or
                // unload five gigabytes of model. Hide instead and keep running.
                api.prevent_close();
                let _ = window.hide();
                let app = window.app_handle();
                let state: tauri::State<'_, Arc<AppState>> = app.state();
                let mut told = state
                    .background_notice
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if !*told {
                    *told = true;
                    let _ = app
                        .notification()
                        .builder()
                        .title("Zen is still running")
                        .body("Your session stays open. Open Zen from the tray, or quit it there.")
                        .show();
                }
            }
        })
        .build(tauri::generate_context!())?
        .run(|app, event| {
            if let RunEvent::ExitRequested { api, .. } = event {
                let state: tauri::State<'_, Arc<AppState>> = app.state();
                if !state.exiting.swap(true, Ordering::AcqRel) {
                    api.prevent_exit();
                    let state = Arc::clone(&state);
                    let handle = app.clone();
                    tauri::async_runtime::spawn(async move {
                        state.sessions.close().await;
                        handle.exit(0);
                    });
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> Channel<InvokeResponseBody> {
        Channel::new(|_| Ok(()))
    }

    #[tokio::test]
    async fn simultaneous_attachments_share_exactly_one_engine() {
        let registry = SessionRegistry::default();
        let (first, second) = tokio::join!(registry.claim(channel()), registry.claim(channel()));
        assert!(Arc::ptr_eq(&first.0, &second.0));
        assert_ne!(first.1, second.1);
        assert_eq!(
            usize::from(first.2.is_some()) + usize::from(second.2.is_some()),
            1
        );
    }

    #[tokio::test]
    async fn restart_waits_for_worker_teardown_and_cannot_accept_old_frames() {
        let registry = SessionRegistry::default();
        let (old, old_revision, worker) = registry.claim(channel()).await;
        let (_input, done) = worker.unwrap();
        old.transport.revoke();
        let next = registry.claim(channel());
        tokio::pin!(next);
        assert!(tokio::time::timeout(Duration::from_millis(20), &mut next)
            .await
            .is_err());
        done.send_replace(true);
        let (new, revision, _) = next.await;
        assert!(!Arc::ptr_eq(&old, &new));
        assert_ne!(old_revision, revision);
        assert!(!new.transport.accepts(old_revision));
        assert!(new.transport.accepts(revision));
    }

    #[tokio::test]
    async fn disconnect_returns_only_after_worker_completion() {
        let registry = SessionRegistry::default();
        let (live, _, worker) = registry.claim(channel()).await;
        let (_input, done) = worker.unwrap();
        let closing = registry.close();
        tokio::pin!(closing);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut closing)
                .await
                .is_err()
        );
        assert!(live.transport.status().2);
        done.send_replace(true);
        closing.await;
    }

    #[test]
    fn only_the_embedded_origin_may_navigate_or_request_microphone_access() {
        for allowed in [
            "http://tauri.localhost/",
            "https://tauri.localhost/capture.js",
            "tauri://localhost/",
        ] {
            assert!(trusted_origin(&tauri::Url::parse(allowed).unwrap()));
        }
        for denied in [
            "https://example.com",
            "http://tauri.localhost.evil.test/",
            "http://tauri.localhost:9876/",
            "http://user@tauri.localhost/",
            "http://127.0.0.1/",
        ] {
            assert!(!trusted_origin(&tauri::Url::parse(denied).unwrap()));
        }
    }
}
