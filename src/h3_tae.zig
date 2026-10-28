//! Optional Core ML tiny decoder (taeh3) for sharp H3 step previews on ANE.
const std = @import("std");
const preview = @import("preview.zig");
const jpeg = @import("jpeg.zig");

const c = @cImport({
    @cInclude("h3_tae_preview.h");
});

const ErrBuf = [512]u8;

var session: ?*c.h3_tae_ctx = null;
var session_path: [512]u8 = undefined;
var session_path_len: usize = 0;

pub fn enabled() bool {
    if (envStr("MINIMAX_H3_TAE")) |raw| {
        if (std.mem.eql(u8, raw, "0")) return false;
    }
    return modelPath() != null;
}

fn envStr(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name.ptr) orelse return null;
    const slice = std.mem.sliceTo(raw, 0);
    return if (slice.len == 0) null else slice;
}

/// Path to a `.mlmodelc` directory, or a bundle root containing size variants.
pub fn modelPath() ?[]const u8 {
    return envStr("MINIMAX_H3_TAE_MODEL");
}

fn useAne() bool {
    if (envStr("MINIMAX_H3_TAE_ANE")) |raw| {
        if (std.mem.eql(u8, raw, "0")) return false;
    }
    return true;
}

fn resolveModelPath(allocator: std.mem.Allocator, lat_h: u32, lat_w: u32) ![]const u8 {
    const base = modelPath() orelse return error.NoModel;
    if (std.mem.endsWith(u8, base, ".mlmodelc")) return try allocator.dupe(u8, base);
    const px = @max(lat_h, lat_w) * 16;
    const variant: []const u8 = blk: {
        if (px <= 768) break :blk "768";
        if (px <= 1024) break :blk "1024";
        if (px <= 1280) break :blk "1280";
        if (px <= 1536) break :blk "1536";
        if (px <= 1792) break :blk "1792";
        break :blk "2048";
    };
    return std.fmt.allocPrint(allocator, "{s}/{s}.mlmodelc", .{ base, variant });
}

fn ensureSession(allocator: std.mem.Allocator, lat_h: u32, lat_w: u32) !*c.h3_tae_ctx {
    const path = try resolveModelPath(allocator, lat_h, lat_w);
    defer allocator.free(path);

    if (session) |s| {
        if (session_path_len == path.len and std.mem.eql(u8, session_path[0..session_path_len], path)) {
            return s;
        }
        c.h3_tae_preview_close(s);
        session = null;
        session_path_len = 0;
    }

    if (c.h3_tae_preview_available() == 0) return error.Unavailable;

    var errbuf: ErrBuf = undefined;
    const s = c.h3_tae_preview_open(path.ptr, if (useAne()) 1 else 0, &errbuf, errbuf.len) orelse {
        const msg = std.mem.sliceTo(&errbuf, 0);
        std.log.warn("[h3-tae] open failed: {s}\n", .{msg});
        return error.OpenFailed;
    };
    const copy_len = @min(path.len, session_path.len);
    @memcpy(session_path[0..copy_len], path[0..copy_len]);
    session_path_len = copy_len;
    session = s;
    return s;
}

/// Channel-major `[C,T,H,W]` f32 → JPEG via Core ML decode (one temporal index).
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
    const rc = c.h3_tae_preview_decode_frame(
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

    const o = opts.normalize();
    const resized = try resizeMaxSide(allocator, rgb, @intCast(out_pw), @intCast(out_ph), o.max_side);
    defer if (resized.owned) allocator.free(resized.rgb);

    const jpg = try jpeg.encodeRgb(allocator, resized.rgb, resized.w, resized.h, 72);
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
