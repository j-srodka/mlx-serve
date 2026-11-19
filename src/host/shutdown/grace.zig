//! Bounded SIGTERM drain for in-flight connection threads and step-boundary
//! cancellation probes used by media progress sinks.

const std = @import("std");

/// Conn-thread drain when `--shutdown-persist-secs` is off (0): unchanged bound.
pub const CONN_DRAIN_MS: i64 = 30_000;

/// Process exit when shutdown grace expires with work still unwinding.
pub const EXIT_GRACE_EXPIRED: u8 = 2;

/// Latched by the serve drain or inference join when the shared deadline passes; the caller exits.
pub var shutdown_timed_out = std.atomic.Value(bool).init(false);

pub const InferenceJoinOutcome = enum { complete, latch_timeout };

/// Wedged past deadline → latch only; `processExitCode` is the sole exit-status source (not join-local exit).
pub fn inferenceJoinOutcome(at_deadline: bool, job_done: bool) InferenceJoinOutcome {
    if (job_done) return .complete;
    if (at_deadline) return .latch_timeout;
    return .complete;
}

pub fn processExitCode() u8 {
    if (shutdown_timed_out.load(.acquire)) return EXIT_GRACE_EXPIRED;
    return 0;
}

/// Monotonic deadline for draining live connection threads after cancel.
/// Idle shutdown (`in_flight == false`) returns `now_ms` so the wait is a no-op.
pub fn drainDeadlineMs(now_ms: i64, armed_deadline_ms: i64, in_flight: bool) i64 {
    if (!in_flight) return now_ms;
    if (armed_deadline_ms > now_ms) return armed_deadline_ms;
    if (armed_deadline_ms > 0) return armed_deadline_ms;
    return now_ms + CONN_DRAIN_MS;
}

pub fn graceExpired(deadline_ms: i64, now_ms: i64) bool {
    return deadline_ms > 0 and now_ms >= deadline_ms;
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

test "shutdown grace: idle shutdown does not wait the full conn drain" {
    const now: i64 = 1_000;
    const deadline = drainDeadlineMs(now, now + 45_000, false);
    try std.testing.expectEqual(now, deadline);
    try std.testing.expect(!graceExpired(deadline, now - 1));
}

test "shutdown grace: in-flight shutdown uses the armed flush deadline" {
    const now: i64 = 5_000;
    try std.testing.expectEqual(@as(i64, 50_000), drainDeadlineMs(now, 50_000, true));
    try std.testing.expect(!graceExpired(50_000, 49_999));
    try std.testing.expect(graceExpired(50_000, 50_000));
}

test "shutdown grace: in-flight with persist off keeps the conn-drain bound" {
    const now: i64 = 2_000;
    try std.testing.expectEqual(now + CONN_DRAIN_MS, drainDeadlineMs(now, 0, true));
}

var grace_test_cancel = std.atomic.Value(bool).init(false);

fn graceTestCancel() bool {
    return grace_test_cancel.load(.acquire);
}

var grace_steps_for_cancel: u32 = 0;
var grace_stop_after: u32 = 0;

fn graceTestCancelAfterK() bool {
    return grace_test_cancel.load(.acquire) and grace_steps_for_cancel >= grace_stop_after;
}

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

test "shutdown grace: a wedged in-flight job expires at the drain deadline" {
    const deadline = drainDeadlineMs(1_000, 1_500, true);
    try std.testing.expect(!graceExpired(deadline, 1_499));
    try std.testing.expect(graceExpired(deadline, 1_500));
}

test "shutdown grace: wedged at deadline latches status 2 for caller exit not join exit" {
    shutdown_timed_out.store(false, .release);
    try std.testing.expectEqual(@as(u8, 0), processExitCode());
    try std.testing.expectEqual(.latch_timeout, inferenceJoinOutcome(true, false));
    shutdown_timed_out.store(true, .release);
    try std.testing.expectEqual(EXIT_GRACE_EXPIRED, processExitCode());
    try std.testing.expectEqual(.complete, inferenceJoinOutcome(false, true));
    shutdown_timed_out.store(false, .release);
}

test "shutdown grace: drainDeadlineMs is the inference join bound not flush plus slack" {
    const start: i64 = 1_000;
    const armed: i64 = 46_000;
    const join_until = drainDeadlineMs(start, armed, true);
    try std.testing.expectEqual(armed, join_until);
    try std.testing.expect(join_until < armed + CONN_DRAIN_MS);
}