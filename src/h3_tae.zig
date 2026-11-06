//! Optional Core ML tiny decoder (taeh3) for sharp H3 step previews on ANE.
const std = @import("std");
const preview = @import("preview.zig");
const jpeg = @import("jpeg.zig");

const h3_tae_ctx = opaque {};

extern fn h3_tae_preview_available() c_int;
extern fn h3_tae_preview_open(
    bundle_dir: [*:0]const u8,
    use_ane: c_int,
    err_out: [*]u8,
    error_size: usize,
) ?*h3_tae_ctx;
extern fn h3_tae_preview_close(ctx: ?*h3_tae_ctx) void;
extern fn h3_tae_preview_decode_frame(
    ctx: ?*h3_tae_ctx,
    latent: [*]const f32,
    t: c_int,
    h: c_int,
    w: c_int,
    c_ch: c_int,
    t_idx: c_int,
    out_rgb: [*]u8,
    out_pw: *c_int,
    out_ph: *c_int,
    err_out: [*]u8,
    error_size: usize,
) c_int;

const ErrBuf = [512]u8;

pub const JpegTimings = struct {
    coreml_ms: u64 = 0,
    jpeg_ms: u64 = 0,
};

var session: ?*h3_tae_ctx = null;
var session_path: [512]u8 = undefined;
var session_path_len: usize = 0;

var open_failed_path: [512]u8 = undefined;
var open_failed_path_len: usize = 0;

var pack_dir: [512]u8 = undefined;
var pack_dir_len: usize = 0;

var model_path_buf: [640]u8 = undefined;
var model_path_len: usize = 0;

/// H3 pack root (directory containing `transformer.safetensors`). Set at generate entry.
pub fn setPackDir(dir: []const u8) void {
    const n = @min(dir.len, pack_dir.len);
    @memcpy(pack_dir[0..n], dir[0..n]);
    pack_dir_len = n;
}

fn envStr(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name.ptr) orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

fn bundleLooksValid(root: []const u8) bool {
    if (root.len == 0 or !std.fs.path.isAbsolute(root)) return false;
    var buf: [640]u8 = undefined;
    const sub = std.fmt.bufPrint(&buf, "{s}/768.mlmodelc", .{root}) catch return false;
    buf[sub.len] = 0;
    if (std.c.access(buf[0..sub.len :0].ptr, 0) != 0) return false;
    return true;
}

fn sharedCacheRoot(out_buf: []u8) ?[]const u8 {
    if (envStr("MINIMAX_H3_TAE_DIR")) |d| return d;
    const home = std.c.getenv("HOME") orelse return null;
    const home_slice = std.mem.span(home);
    const suffix = "/.mlx-serve/shared/minimax_h3_tae";
    if (home_slice.len + suffix.len > out_buf.len) return null;
    @memcpy(out_buf[0..home_slice.len], home_slice);
    @memcpy(out_buf[home_slice.len .. home_slice.len + suffix.len], suffix);
    return out_buf[0 .. home_slice.len + suffix.len];
}

/// Resolved bundle root (directory containing `768.mlmodelc` …), or explicit `.mlmodelc` path.
pub fn modelPath() ?[]const u8 {
    if (envStr("MINIMAX_H3_TAE_MODEL")) |m| return m;

    model_path_len = 0;

    if (pack_dir_len > 0) {
        const n = std.fmt.bufPrint(&model_path_buf, "{s}/minimax_h3_tae", .{pack_dir[0..pack_dir_len]}) catch return null;
        if (bundleLooksValid(n)) {
            model_path_len = n.len;
            return model_path_buf[0..model_path_len];
        }
    }

    if (sharedCacheRoot(&model_path_buf)) |shared| {
        if (bundleLooksValid(shared)) {
            model_path_len = shared.len;
            return model_path_buf[0..model_path_len];
        }
    }

    return null;
}

pub fn enabled() bool {
    if (envStr("MINIMAX_H3_TAE")) |raw| {
        if (std.mem.eql(u8, raw, "0")) return false;
    }
    if (h3_tae_preview_available() == 0) return false;
    return modelPath() != null;
}

fn useAne() bool {
    if (envStr("MINIMAX_H3_TAE_ANE")) |raw| {
        if (std.mem.eql(u8, raw, "0")) return false;
    }
    return true;
}

fn variantName(lat_h: u32, lat_w: u32) []const u8 {
    const px = @max(lat_h, lat_w) * 16;
    if (px <= 768) return "768";
    if (px <= 1024) return "1024";
    if (px <= 1280) return "1280";
    if (px <= 1536) return "1536";
    if (px <= 1792) return "1792";
    return "2048";
}

fn resolveModelPath(allocator: std.mem.Allocator, lat_h: u32, lat_w: u32) ![:0]const u8 {
    const base = modelPath() orelse return error.NoModel;
    if (std.mem.endsWith(u8, base, ".mlmodelc")) return try allocator.dupeSentinel(u8, base, 0);
    const variant = variantName(lat_h, lat_w);
    const tmp = try std.fmt.allocPrint(allocator, "{s}/{s}.mlmodelc", .{ base, variant });
    defer allocator.free(tmp);
    return try allocator.dupeSentinel(u8, tmp, 0);
}

fn ensureSession(allocator: std.mem.Allocator, lat_h: u32, lat_w: u32) !*h3_tae_ctx {
    const path = try resolveModelPath(allocator, lat_h, lat_w);
    defer allocator.free(path);

    if (session) |s| {
        if (session_path_len == path.len and std.mem.eql(u8, session_path[0..session_path_len], path)) {
            return s;
        }
        h3_tae_preview_close(s);
        session = null;
        session_path_len = 0;
    }

    if (open_failed_path_len > 0 and open_failed_path_len == path.len and
        std.mem.eql(u8, open_failed_path[0..open_failed_path_len], path))
    {
        return error.OpenFailed;
    }

    if (h3_tae_preview_available() == 0) return error.Unavailable;

    var errbuf: ErrBuf = undefined;
    const s = h3_tae_preview_open(path.ptr, if (useAne()) 1 else 0, &errbuf, errbuf.len) orelse {
        const msg = std.mem.sliceTo(&errbuf, 0);
        std.log.warn("[h3-tae] open failed path={s}: {s}\n", .{ path, msg });
        const copy_len = @min(path.len, open_failed_path.len);
        @memcpy(open_failed_path[0..copy_len], path[0..copy_len]);
        open_failed_path_len = copy_len;
        return error.OpenFailed;
    };
    open_failed_path_len = 0;
    const copy_len = @min(path.len, session_path.len);
    @memcpy(session_path[0..copy_len], path[0..copy_len]);
    session_path_len = copy_len;
    session = s;
    return s;
}

/// Channel-major `[C,T,H,W]` f32 (MLX `[1,C,T,H,W]` eval order) → JPEG via Core ML.
pub fn jpegFromLatent(
    allocator: std.mem.Allocator,
    latent: []const f32,
    t: u32,
    h: u32,
    w: u32,
    c_ch: u32,
    t_idx: u32,
    opts: preview.Opts,
) !preview.Encoded {
    return jpegFromLatentProfiled(allocator, latent, t, h, w, c_ch, t_idx, opts, null, null);
}

pub fn jpegFromLatentProfiled(
    allocator: std.mem.Allocator,
    latent: []const f32,
    t: u32,
    h: u32,
    w: u32,
    c_ch: u32,
    t_idx: u32,
    opts: preview.Opts,
    io: ?std.Io,
    timings: ?*JpegTimings,
) !preview.Encoded {
    const s = try ensureSession(allocator, h, w);

    const nhwc_len = @as(usize, t) * @as(usize, h) * @as(usize, w) * @as(usize, c_ch);
    const nhwc = try allocator.alloc(f32, nhwc_len);
    defer allocator.free(nhwc);

    const hw: usize = @as(usize, h) * @as(usize, w);
    const thw: usize = @as(usize, t) * hw;
    for (0..c_ch) |ci| {
        for (0..t) |ti| {
            var p: usize = 0;
            while (p < hw) : (p += 1) {
                const y = p / @as(usize, w);
                const x = p % @as(usize, w);
                const dst = ((ti * @as(usize, h) + y) * @as(usize, w) + x) * @as(usize, c_ch) + ci;
                nhwc[dst] = latent[ci * thw + ti * hw + p];
            }
        }
    }

    const rgb_len = @as(usize, h) * @as(usize, w) * 256 * 3;
    const rgb = try allocator.alloc(u8, rgb_len);
    defer allocator.free(rgb);

    var errbuf: ErrBuf = undefined;
    var out_pw: c_int = 0;
    var out_ph: c_int = 0;

    var prof_mark: u64 = 0;
    var prof_start: ?std.Io.Timestamp = null;
    if (timings != null and io != null) prof_start = std.Io.Timestamp.now(io.?, .boot);

    const rc = h3_tae_preview_decode_frame(
        s,
        nhwc.ptr,
        @intCast(t),
        @intCast(h),
        @intCast(w),
        @intCast(c_ch),
        @intCast(t_idx),
        rgb.ptr,
        &out_pw,
        &out_ph,
        &errbuf,
        errbuf.len,
    );
    if (rc != 0) {
        const msg = std.mem.sliceTo(&errbuf, 0);
        std.log.warn("[h3-tae] decode failed: {s}\n", .{msg});
        return error.DecodeFailed;
    }

    if (timings) |tm| {
        if (prof_start) |ps| {
            const io_ref = io.?;
            const cum: u64 = @intCast(ps.untilNow(io_ref, .boot).nanoseconds);
            tm.coreml_ms = (cum - prof_mark) / 1_000_000;
            prof_mark = cum;
        }
    }

    const o = opts.normalize();
    const resized = try resizeMaxSide(allocator, rgb, @intCast(out_pw), @intCast(out_ph), o.max_side);
    defer if (resized.owned) allocator.free(resized.rgb);

    const jpg = try jpeg.encodeRgb(allocator, resized.rgb, resized.w, resized.h, 72);

    if (timings) |tm| {
        if (prof_start) |ps| {
            const io_ref = io.?;
            const cum: u64 = @intCast(ps.untilNow(io_ref, .boot).nanoseconds);
            tm.jpeg_ms = (cum - prof_mark) / 1_000_000;
        }
    }

    return .{ .jpeg = jpg, .w = resized.w, .h = resized.h };
}

const Resize = struct {
    rgb: []const u8,
    w: u32,
    h: u32,
    owned: bool,
};

fn resizeMaxSide(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32, max_side: u32) !Resize {
    if (max_side == 0) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const long: u32 = @max(w, h);
    if (long <= max_side) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const nw: u32 = @max(1, (w * max_side) / long);
    const nh: u32 = @max(1, (h * max_side) / long);
    const out = try allocator.alloc(u8, @as(usize, nw) * @as(usize, nh) * 3);
    bilinear(rgb, w, h, out, nw, nh);
    return .{ .rgb = out, .w = nw, .h = nh, .owned = true };
}

fn bilinear(src: []const u8, sw: u32, sh: u32, dst: []u8, dw: u32, dh: u32) void {
    const x_scale = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw));
    const y_scale = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh));
    var y: u32 = 0;
    while (y < dh) : (y += 1) {
        const fy = (@as(f32, @floatFromInt(y)) + 0.5) * y_scale - 0.5;
        const y0 = @as(u32, @intFromFloat(@max(fy, 0)));
        const y1 = @min(y0 + 1, sh - 1);
        const ty = fy - @as(f32, @floatFromInt(y0));
        var x: u32 = 0;
        while (x < dw) : (x += 1) {
            const fx = (@as(f32, @floatFromInt(x)) + 0.5) * x_scale - 0.5;
            const x0 = @as(u32, @intFromFloat(@max(fx, 0)));
            const x1 = @min(x0 + 1, sw - 1);
            const tx = fx - @as(f32, @floatFromInt(x0));
            var ch: u32 = 0;
            while (ch < 3) : (ch += 1) {
                const a = samp(src, sw, x0, y0, ch);
                const b = samp(src, sw, x1, y0, ch);
                const c0 = samp(src, sw, x0, y1, ch);
                const d = samp(src, sw, x1, y1, ch);
                const top = a + (b - a) * tx;
                const bot = c0 + (d - c0) * tx;
                const v = top + (bot - top) * ty;
                dst[(@as(usize, y) * @as(usize, dw) + @as(usize, x)) * 3 + ch] = @intFromFloat(@min(@max(v, 0), 255) + 0.5);
            }
        }
    }
}

fn samp(src: []const u8, sw: u32, x: u32, y: u32, ch: u32) f32 {
    return @floatFromInt(src[(@as(usize, y) * @as(usize, sw) + @as(usize, x)) * 3 + ch]);
}

test "h3 tae variant bucket for 704px canvas latent grid" {
    const lat_h: u32 = 44;
    const lat_w: u32 = 44;
    const px: u32 = @as(u32, @max(lat_h, lat_w)) * 16;
    try std.testing.expectEqual(@as(u32, 704), px);
    try std.testing.expect(px <= 768);
    try std.testing.expectEqualStrings("768", variantName(lat_h, lat_w));
}

fn bumpStackForTest() void {
    var x: [8192]u8 = undefined;
    @memset(&x, 0xA5);
}

test "h3 tae modelPath uses module buffer not caller stack" {
    const base = modelPath() orelse return;
    bumpStackForTest();
    const again = modelPath() orelse return;
    try std.testing.expectEqualStrings(base, again);
    try std.testing.expect(base.ptr == again.ptr);
}

test "h3 tae resolve path is nul terminated with variant suffix" {
    const a = std.testing.allocator;
    _ = modelPath() orelse return;
    const path = try resolveModelPath(a, 44, 44);
    defer a.free(path);
    try std.testing.expect(path[path.len] == 0);
    try std.testing.expect(std.mem.endsWith(u8, path, "/768.mlmodelc"));
}
