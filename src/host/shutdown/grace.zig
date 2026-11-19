//! Bounded SIGTERM drain: one deadline captured at shutdown, enforced by a watchdog.

const std = @import("std");
const crt = @import("../../sdk/util/crt.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const log = @import("../../sdk/util/log.zig");

/// Conn-thread drain when `--shutdown-persist-secs` is off (0): bound from SIGTERM.
pub const CONN_DRAIN_MS: i64 = 30_000;

/// Process exit when the grace watchdog fires.
pub const EXIT_GRACE_EXPIRED: u8 = 2;

/// Monotonic grace end, set once when shutdown begins on the serve thread.
pub var shutdown_deadline_ms = std.atomic.Value(i64).init(0);

/// Clean path latched before `serve` returns; the watchdog exits quietly when set.
pub var shutdown_clean = std.atomic.Value(bool).init(false);

var watchdog_started = std.atomic.Value(bool).init(false);
var watchdog_thread: ?std.Thread = null;

/// Pure: one horizon from the SIGTERM instant — not recomputed from a later `now`.
pub fn captureDeadlineMs(sigterm_now_ms: i64, persist_secs: u32) i64 {
    if (persist_secs > 0) return sigterm_now_ms + @as(i64, persist_secs) * 1000;
    return sigterm_now_ms + CONN_DRAIN_MS;
}

pub fn armDeadlineOnce(sigterm_now_ms: i64, persist_secs: u32) i64 {
    const deadline = captureDeadlineMs(sigterm_now_ms, persist_secs);
    if (shutdown_deadline_ms.cmpxchgStrong(0, deadline, .acq_rel, .acquire)) |_| return deadline;
    return shutdown_deadline_ms.load(.acquire);
}

pub fn graceExpired(deadline_ms: i64, now_ms: i64) bool {
    return deadline_ms > 0 and now_ms >= deadline_ms;
}

pub fn syncFlushDeadline(slot: *std.atomic.Value(i64), deadline_ms: i64) void {
    if (deadline_ms <= 0) return;
    _ = slot.cmpxchgStrong(0, deadline_ms, .acq_rel, .acquire);
}

pub fn resetForServe() void {
    shutdown_deadline_ms.store(0, .monotonic);
    shutdown_clean.store(false, .release);
    watchdog_started.store(false, .release);
    watchdog_thread = null;
}

/// Capture the deadline and start the grace watchdog (idempotent).
pub fn beginShutdown(io: std.Io, sigterm_now_ms: i64, persist_secs: u32) i64 {
    const deadline = armDeadlineOnce(sigterm_now_ms, persist_secs);
    if (watchdog_started.swap(true, .acq_rel)) return deadline;
    watchdog_thread = std.Thread.spawn(.{}, watchdogLoop, .{io, deadline}) catch |err| {
        log.err("shutdown: watchdog spawn failed ({s}); grace enforced only on the serve thread\n", .{@errorName(err)});
        watchdog_started.store(false, .release);
        return deadline;
    };
    return deadline;
}

fn watchdogLoop(io: std.Io, deadline_ms: i64) void {
    while (true) {
        if (shutdown_clean.load(.acquire)) return;
        const now = io_util.nowMsMonotonic(io);
        if (graceExpired(deadline_ms, now)) break;
        crt.sleepMs(20);
    }
    if (shutdown_clean.load(.acquire)) return;
    log.err("shutdown: grace deadline {d}ms reached; exiting with status {d}\n", .{ deadline_ms, EXIT_GRACE_EXPIRED });
    std.process.exit(EXIT_GRACE_EXPIRED);
}

pub fn markCleanShutdown() void {
    shutdown_clean.store(true, .release);
    if (watchdog_thread) |t| {
        t.join();
        watchdog_thread = null;
    }
}

pub fn cleanShutdownExitCode(clean_before_deadline: bool, at_deadline: bool) u8 {
    if (clean_before_deadline) return 0;
    if (at_deadline) return EXIT_GRACE_EXPIRED;
    return 0;
}

/// Media-style loop: check `cancelled` between steps (after each increment).
pub fn runSteps(total: u32, steps_done: *u32, cancelled: *const fn () bool) error{Cancelled}!void {
    steps_done.* = 0;
    var step: u32 = 0;
    while (step < total) : (step += 1) {
        if (cancelled()) return error.Cancelled;
        steps_done.* = step + 1;
    }
}

test "shutdown grace: captureDeadline uses persist or conn drain from sigterm only" {
    try std.testing.expectEqual(@as(i64, 55_000), captureDeadlineMs(10_000, 45));
    try std.testing.expectEqual(@as(i64, 40_000), captureDeadlineMs(10_000, 0));
}

test "shutdown grace: deadline captured once is identical for all waiters" {
    const sigterm: i64 = 10_000;
    const d1 = captureDeadlineMs(sigterm, 45);
    const d2 = captureDeadlineMs(sigterm, 45);
    try std.testing.expectEqual(d1, d2);
    // Second-horizon defect: a later sigterm instant must not become the shared deadline.
    try std.testing.expect(d1 != captureDeadlineMs(sigterm + 60_000, 45));
}

test "shutdown grace: clean latch before deadline yields exit 0 not watchdog 2" {
    try std.testing.expectEqual(@as(u8, 0), cleanShutdownExitCode(true, false));
    try std.testing.expectEqual(@as(u8, 0), cleanShutdownExitCode(true, true));
    try std.testing.expectEqual(EXIT_GRACE_EXPIRED, cleanShutdownExitCode(false, true));
    try std.testing.expectEqual(@as(u8, 0), cleanShutdownExitCode(false, false));
}

var grace_test_cancel = std.atomic.Value(bool).init(false);

fn graceTestCancelAfterK() bool {
    return grace_test_cancel.load(.acquire) and grace_steps_for_cancel >= grace_stop_after;
}

var grace_steps_for_cancel: u32 = 0;
var grace_stop_after: u32 = 0;

test "shutdown grace: cooperative render stops when cancelled between steps" {
    const k: u32 = 5;
    grace_test_cancel.store(false, .release);
    grace_stop_after = k + 1;
    grace_steps_for_cancel = 0;
    try runSteps(5, &grace_steps_for_cancel, graceTestCancelAfterK);
    try std.testing.expectEqual(k, grace_steps_for_cancel);
    grace_test_cancel.store(true, .release);
    grace_steps_for_cancel = 0;
    try std.testing.expectError(error.Cancelled, runSteps(100, &grace_steps_for_cancel, graceTestCancelAfterK));
    try std.testing.expectEqual(k + 1, grace_steps_for_cancel);
    grace_test_cancel.store(false, .release);
    grace_stop_after = 0;
}

test "shutdown grace: graceExpired on the captured timestamp" {
    const deadline = captureDeadlineMs(1_000, 0);
    try std.testing.expect(!graceExpired(deadline, deadline - 1));
    try std.testing.expect(graceExpired(deadline, deadline));
}