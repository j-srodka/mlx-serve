//! GenJob runners; handleGen

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/media.zig");
const scheduler_mod = @import("../gpu_thread/requests.zig");
const model_registry_mod = @import("../registry/registry.zig");
const media_mod = @import("gen.zig");
const plugin_http = @import("plugin/http.zig");
const PluginEngine = @import("plugin/engine.zig").Engine;
const http_server = @import("../http/server.zig");
const minimax_h3_plugin = @import("../../plugins/minimax_h3/plugin.zig");

const Conn = @import("../http/conn.zig").Conn;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;

/// Media-generation job payload. Carries the connection + body + resolved
/// model into the inference thread, where `genJobRun` runs the actual
/// generation (mlx + SSE writes) on the GPU-stream-owning thread.
const GenJob = struct {
    allocator: std.mem.Allocator,
    conn: *Conn,
    body: []const u8,
    lm: *model_registry_mod.LoadedModel,
    route: media_mod.GenRoute,
};

/// Inference-thread entry point for a gen job. Dispatches on the engine slot
/// and writes the full HTTP/SSE response to the (parked) connection.
fn genJobRun(ctx: *anyopaque) void {
    const job: *GenJob = @ptrCast(@alignCast(ctx));
    if (job.route == .decisions) return genJobRunMany(&.{ctx});
    const p = job.lm.media_plugin orelse return;
    if (job.route == .h3_refmods) {
        refmodsJobRun(job) catch |err| {
            log.warn("[gen] {s} job failed: {s}\n", .{ @tagName(job.route), @errorName(err) });
        };
        return;
    }
    plugin_http.handle(job.allocator, job.conn, job.body, p, job.route, job.lm.id) catch |err| {
        log.warn("[gen] {s} job failed: {s}\n", .{ @tagName(job.route), @errorName(err) });
    };
}

fn refmodsJobRun(job: *GenJob) !void {
    const engine = job.lm.media_plugin orelse return;
    if (!std.mem.eql(u8, engine.table.name.slice(), "minimax_h3")) {
        try sendErrorResponse(job.allocator, job.conn, "400 Bad Request", "invalid_request_error", "POST /v1/video/refmods requires a loaded MiniMax-H3 (minimax_h3) model", 400);
        return;
    }
    var msg_buf: [512]u8 = undefined;
    var msg_len: usize = 0;
    const req = abi.Request{
        .body = .of(job.body),
        .model_id = .of(job.lm.id),
        .message = .{ .ptr = &msg_buf, .cap = msg_buf.len, .len = &msg_len },
        .dir = &engine.pack,
        .machine = &engine.machine,
    };
    const v = minimax_h3_plugin.refmodsExtract(engine.handle, &req);
    if (v.isOk()) {
        const json = minimax_h3_plugin.refmodsResultTake(engine.handle);
        defer std.heap.c_allocator.free(json);
        return media_mod.sendBytesJson(job.conn, job.allocator, json);
    }
    const text: []const u8 = if (msg_len > 0) msg_buf[0..msg_len] else v.why.slice();
    log.warn("[gen] refmods refused: {s}\n", .{text});
    const status: u16 = switch (v.code) {
        .unsupported => 400,
        .too_large => 400,
        .busy => 503,
        .failed => media_mod.jobFailureStatus(error.RefmodsFailed),
        else => 500,
    };
    try media_mod.sendError(job.conn, status, text);
}

/// Inference-thread entry point for decision jobs queued back to back for one
/// model (`GenRequest.merge`): they are answered in one pass.
fn genJobRunMany(ctxs: []const *anyopaque) void {
    const first: *GenJob = @ptrCast(@alignCast(ctxs[0]));
    const p = first.lm.media_plugin orelse return;
    var queued: [scheduler_mod.MAX_MERGED_WEIGHT]plugin_http.Queued = undefined;
    for (ctxs, queued[0..ctxs.len]) |c, *q| {
        const job: *GenJob = @ptrCast(@alignCast(c));
        q.* = .{ .allocator = job.allocator, .conn = job.conn, .body = job.body };
    }
    plugin_http.handleMany(p, first.lm.id, queued[0..ctxs.len]);
}

/// Dispatch a media-generation request to the inference thread. `lm` is the
/// already-resolved + refcounted model (so it can't be evicted mid-gen). The
/// generation runs on the scheduler's inference thread (the sole mlx caller);
/// the body writes its own response. A wrong-modality target gets a clear 400.
pub fn handleGen(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *model_registry_mod.LoadedModel, route: media_mod.GenRoute) !void {
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    if (!PluginEngine.serves(lm.media_plugin, route.modality())) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Target model does not support this modality. Load the matching image/audio/video/3D/decision model and target it by id.", 400);
        return;
    }
    var job = GenJob{ .allocator = allocator, .conn = stream, .body = body, .lm = lm, .route = route };
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = genJobRun, .model = lm, .decision = route == .decisions };
    if (lm.media_plugin) |p| req.peak_bytes = if (route == .h3_refmods) 0 else p.requestPeak(body);
    if (route == .decisions) if (lm.media_plugin) |p| {
        req.merge = .{ .run_many = genJobRunMany, .weight = plugin_http.questionCount(body), .window_us = p.batch_window_us };
    };
    scheduler.runGeneration(&req) catch |err| switch (err) {
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        else => return err,
    };
    // The job body wrote the full HTTP/SSE response on the inference thread.
}
