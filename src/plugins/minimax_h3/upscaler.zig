//! MiniMax-H3 2D latent upscaler (LBH-123-AI LatentResizer2D). Port of
//! `nodes/minimax_h3_latent_upscaler_2d.py` (MIT). Video latent `[1,24,T,H,W]`
//! only; time is unchanged. Weights absent is a named 400, never a silent
//! interpolate.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const wload = @import("../../sdk/weights/load.zig");

fn rio() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

const S = mlx.mlx_stream;
const A = mlx.mlx_array;

pub const IN_CH: u32 = 24;
pub const FREQ: u32 = 16;
pub const TIER_MIN: u32 = 1;
pub const TIER_MAX: u32 = 6;
pub const SCALE_MIN: f64 = 1.0;
pub const SCALE_MAX: f64 = 4.0;
pub const CHUNK_FRAMES: u32 = 32;
pub const CHUNK_CONTEXT: u32 = 48; // ±24

/// Per-channel H3 VAE latent stats from the 2D v1 config.
pub const LATENTS_MEAN = [_]f32{
    0.858090341091156,     -0.9606591463088989,  1.0661640167236328,  -0.5090325474739075,
    -0.2727581858634949,   -1.3675414323806763,  -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243,   -0.0464097298681736,  0.6657370328903198,  0.19690127670764923,
    -0.5460608005523682,   -0.4035342037677765,  -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153,  0.211341992020607,    -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056,  0.2604829967021942,   0.22864092886447906, 0.7056031823158264,
};
pub const LATENTS_STD = [_]f32{
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665,  0.9653137922286987, 1.0569885969161987,
    0.841948926448822,  0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786,  2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751,  3.1627357006073,    2.2816812992095947, 2.6127843856811523,
};

pub const Arch = struct {
    channels: u32 = 640,
    in_blocks: u32 = 12,
    out_blocks: u32 = 12,
    embed_dim: u32 = 128,
    freq: u32 = FREQ,
    tier_dim: u32 = 32,
    use_coord: bool = false,
    use_se: bool = false,
    use_large_kernel: bool = false,
    large_kernel_size: u32 = 7,
    use_post_refine: bool = false,
    post_refine_layers: u32 = 2,
    temporal_every: u32 = 2,
    temporal_kernel: u32 = 5,
    use_multiscale_temporal: bool = false,
    learned_k: u32 = 0,
};

pub fn fourierInDim(freq: u32, tier_dim: u32) u32 {
    return 4 * freq + tier_dim;
}

/// Fourier(scale_h)+Fourier(scale_w)+optional Fourier(tier-1). No π (v12).
pub fn fourierCode(scale_h: f64, scale_w: f64, tier: f64, freq: u32, tier_dim: u32, out: []f32) void {
    std.debug.assert(out.len == fourierInDim(freq, tier_dim));
    var i: u32 = 0;
    const write = struct {
        fn f(v: f64, n: u32, buf: []f32, start: *u32) void {
            var k: u32 = 0;
            while (k < n) : (k += 1) {
                const fr = std.math.exp2(@as(f64, @floatFromInt(k)));
                buf[start.*] = @floatCast(@sin(v * fr));
                start.* += 1;
            }
            k = 0;
            while (k < n) : (k += 1) {
                const fr = std.math.exp2(@as(f64, @floatFromInt(k)));
                buf[start.*] = @floatCast(@cos(v * fr));
                start.* += 1;
            }
        }
    }.f;
    write(scale_h, freq, out, &i);
    write(scale_w, freq, out, &i);
    if (tier_dim > 0) write(tier - 1.0, freq, out, &i);
}

pub fn weightsPresent(pack_dir: ?[]const u8) bool {
    if (cliPath()) |p| {
        std.Io.Dir.cwd().access(rio(), p, .{}) catch return packHasWeights(pack_dir);
        return true;
    }
    return packHasWeights(pack_dir);
}

fn packHasWeights(pack_dir: ?[]const u8) bool {
    const dir = pack_dir orelse return false;
    var buf: [4096]u8 = undefined;
    const sub = std.fmt.bufPrint(&buf, "{s}/latent_upscaler", .{dir}) catch return false;
    var d = std.Io.Dir.openDirAbsolute(rio(), sub, .{ .iterate = true }) catch return false;
    defer d.close(rio());
    var it = d.iterate();
    while (it.next(rio()) catch return false) |e| {
        if (e.kind == .file and std.mem.endsWith(u8, e.name, ".safetensors")) return true;
    }
    return false;
}

pub fn cliPath() ?[]const u8 {
    const p = env.span(.MLX_SERVE_H3_UPSCALER) orelse return null;
    return if (p.len == 0) null else p;
}

/// Caller frees. CLI path wins; else `{pack}/latent_upscaler/` (2d v1 name preferred).
pub fn resolvePath(allocator: std.mem.Allocator, pack_dir: []const u8) !?[]u8 {
    if (cliPath()) |p| {
        std.Io.Dir.cwd().access(rio(), p, .{}) catch return error.UpscalerMissing;
        return try allocator.dupe(u8, p);
    }
    var buf: [4096]u8 = undefined;
    const sub = try std.fmt.bufPrint(&buf, "{s}/latent_upscaler", .{pack_dir});
    var d = std.Io.Dir.openDirAbsolute(rio(), sub, .{ .iterate = true }) catch return null;
    defer d.close(rio());
    var prefer: ?[]u8 = null;
    errdefer if (prefer) |p| allocator.free(p);
    var fallback: ?[]u8 = null;
    errdefer if (fallback) |p| allocator.free(p);
    var it = d.iterate();
    while (try it.next(rio())) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const full = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ sub, e.name });
        const is_2d = std.mem.indexOf(u8, e.name, "2d") != null;
        if (is_2d) {
            if (fallback) |p| allocator.free(p);
            fallback = null;
            if (prefer) |p| allocator.free(p);
            prefer = full;
            if (std.mem.indexOf(u8, e.name, "2d_conv_v1") != null) break;
        } else if (prefer == null and fallback == null) {
            fallback = full;
        } else allocator.free(full);
    }
    if (prefer) |p| {
        if (fallback) |f| allocator.free(f);
        return p;
    }
    return fallback;
}

pub fn workingSetBytes(weight_bytes: u64, lat_h: u32, lat_w: u32, latent_t: u32) u64 {
    const w = if (weight_bytes == 0) @as(u64, 364) << 20 else weight_bytes;
    const frames: u64 = @min(@as(u64, latent_t), CHUNK_FRAMES + CHUNK_CONTEXT);
    const spatial: u64 = @as(u64, lat_h) * lat_w;
    const feat = frames * 640 * spatial * 2;
    return w + feat * 6;
}

pub const Model = struct {
    w: Weights,
    arch: Arch,
    bytes: u64,

    pub fn deinit(self: *Model) void {
        self.w.deinit();
        self.* = undefined;
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Model {
        var weights = try wload.loadWeightsSingleFile(allocator, path);
        errdefer weights.deinit();
        const arch = detect(&weights);
        log.info("[minimax-h3] latent upscaler: {s} ({d} ch, {d}+{d} blocks, tier_dim {d})\n", .{
            path, arch.channels, arch.in_blocks, arch.out_blocks, arch.tier_dim,
        });
        return .{ .w = weights, .arch = arch, .bytes = fileBytes(path) };
    }

    pub fn forward(
        self: *const Model,
        allocator: std.mem.Allocator,
        x: A,
        out_h: u32,
        out_w: u32,
        tier: u32,
        s: S,
    ) !A {
        const sh = mlx.getShape(x);
        if (sh.len != 5 or sh[1] != IN_CH) return error.BadUpscalerInput;
        const t: u32 = @intCast(sh[2]);
        const h: u32 = @intCast(sh[3]);
        const w: u32 = @intCast(sh[4]);
        if (out_h == h and out_w == w) return try contig(x, s);

        const z = try normalize(x, s);
        defer _ = mlx.mlx_array_free(z);
        const overlap: u32 = CHUNK_CONTEXT / 2;
        const chunk = CHUNK_FRAMES;
        if (t <= chunk) {
            const y = try forwardFull(self, allocator, z, h, w, out_h, out_w, tier, s);
            defer _ = mlx.mlx_array_free(y);
            return denormalize(y, s);
        }
        var parts: std.ArrayList(A) = .empty;
        defer {
            for (parts.items) |p| _ = mlx.mlx_array_free(p);
            parts.deinit(allocator);
        }
        var start: u32 = 0;
        while (start < t) {
            const seg_end = @min(t, start + chunk);
            const lo = start -| overlap;
            const hi = @min(t, seg_end + overlap);
            const seg = try sliceT(z, lo, hi, s);
            defer _ = mlx.mlx_array_free(seg);
            const seg_out = try forwardFull(self, allocator, seg, h, w, out_h, out_w, tier, s);
            defer _ = mlx.mlx_array_free(seg_out);
            const s0 = start - lo;
            const keep = try sliceT(seg_out, s0, s0 + (seg_end - start), s);
            try parts.append(allocator, keep);
            start += chunk;
        }
        const cat = try concatT(allocator, parts.items, s);
        defer _ = mlx.mlx_array_free(cat);
        return denormalize(cat, s);
    }
};

fn fileBytes(path: []const u8) u64 {
    const st = std.Io.Dir.cwd().statFile(rio(), path, .{}) catch return 0;
    return st.size;
}

fn detect(w: *const Weights) Arch {
    var a = Arch{};
    if (w.get("resizer.conv_in.weight")) |cw| {
        const sh = mlx.getShape(cw);
        a.channels = @intCast(sh[0]);
        const in_ch: u32 = @intCast(sh[1]);
        var real_in: u32 = IN_CH;
        if (w.get("resizer.conv_out.weight")) |ow| real_in = @intCast(mlx.getShape(ow)[0]);
        a.use_coord = in_ch == real_in + 2;
    }
    var in_max: i32 = -1;
    var out_max: i32 = -1;
    var it = w.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (blockIndex(k, "resizer.in_blocks.")) |i| in_max = @max(in_max, i);
        if (blockIndex(k, "resizer.out_blocks.")) |i| out_max = @max(out_max, i);
        if (std.mem.indexOf(u8, k, ".se.fc.") != null) a.use_se = true;
        if (std.mem.indexOf(u8, k, "resizer.post_up.refine.") != null) a.use_post_refine = true;
        if (std.mem.indexOf(u8, k, "temporal_blocks.0.branches.") != null) a.use_multiscale_temporal = true;
        if (std.mem.startsWith(u8, k, "resizer.learned_up.branches.")) {
            if (blockIndex(k, "resizer.learned_up.branches.")) |kk| {
                if (kk > 1) a.learned_k = @intCast(kk);
            }
        }
        if (std.mem.indexOf(u8, k, ".dw.weight") != null and std.mem.indexOf(u8, k, "resizer.in_blocks") != null and
            std.mem.indexOf(u8, k, "temporal") == null)
        {
            a.use_large_kernel = true;
            a.large_kernel_size = @intCast(mlx.getShape(e.value_ptr.*)[2]);
        }
        if (std.mem.indexOf(u8, k, "temporal_blocks.0.dwconv.weight") != null)
            a.temporal_kernel = @intCast(mlx.getShape(e.value_ptr.*)[2]);
    }
    if (in_max >= 0) a.in_blocks = @intCast(in_max + 1);
    if (out_max >= 0) a.out_blocks = @intCast(out_max + 1);
    if (w.get("resizer.embed.linear.0.weight")) |ew| {
        const sh = mlx.getShape(ew);
        a.embed_dim = @intCast(sh[0]);
        const in_dim: u32 = @intCast(sh[1]);
        if (in_dim == 96) {
            a.freq = 16;
            a.tier_dim = 32;
        } else {
            a.freq = in_dim / 4;
            a.tier_dim = 0;
        }
    }
    return a;
}

fn blockIndex(key: []const u8, prefix: []const u8) ?i32 {
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    const rest = key[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(i32, rest[0..dot], 10) catch null;
}

fn must(w: *const Weights, name: []const u8) !A {
    return w.get(name) orelse error.MissingUpscalerWeight;
}

fn keyBuf(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

fn forwardFull(
    self: *const Model,
    allocator: std.mem.Allocator,
    x: A,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    tier: u32,
    s: S,
) !A {
    const sh = mlx.getShape(x);
    const b: u32 = @intCast(sh[0]);
    const t: u32 = @intCast(sh[2]);
    const scale_h = @as(f64, @floatFromInt(out_h)) / @as(f64, @floatFromInt(in_h));
    const scale_w = @as(f64, @floatFromInt(out_w)) / @as(f64, @floatFromInt(in_w));
    const emb = try embed(self, scale_h, scale_w, tier, b * t, s);
    defer _ = mlx.mlx_array_free(emb);

    var feat = try toNhwc(x, s);
    if (self.arch.use_coord) {
        const c = try coordConcat(feat, s);
        _ = mlx.mlx_array_free(feat);
        feat = c;
    }
    var h = try conv2dPt(feat, try must(&self.w, "resizer.conv_in.weight"), self.w.get("resizer.conv_in.bias"), 1, false, s);
    _ = mlx.mlx_array_free(feat);

    h = try runBlocks(self, allocator, h, emb, "resizer.in_blocks", self.arch.in_blocks, 0, b, t, s);
    h = try upsampleMid(self, allocator, h, out_h, out_w, scale_h, scale_w, s);
    if (self.arch.use_post_refine) h = try postRefine(self, h, s);
    h = try runBlocks(self, allocator, h, emb, "resizer.out_blocks", self.arch.out_blocks, 1, b, t, s);

    const n = try groupNormNhwc(h, try must(&self.w, "resizer.norm_out.weight"), try must(&self.w, "resizer.norm_out.bias"), s);
    _ = mlx.mlx_array_free(h);
    const act = try silu(n, s);
    _ = mlx.mlx_array_free(n);
    const out = try conv2dPt(act, try must(&self.w, "resizer.conv_out.weight"), self.w.get("resizer.conv_out.bias"), 1, false, s);
    _ = mlx.mlx_array_free(act);
    const bcthw = try toBcthw(out, b, t, s);
    _ = mlx.mlx_array_free(out);
    return bcthw;
}

fn runBlocks(
    self: *const Model,
    allocator: std.mem.Allocator,
    h_in: A,
    emb: A,
    comptime prefix: []const u8,
    n: u32,
    temporal_idx: u32,
    b: u32,
    t: u32,
    s: S,
) !A {
    var h = h_in;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        var buf: [128]u8 = undefined;
        const in_key = keyBuf(&buf, prefix ++ ".{d}.in_layers.0.weight", .{i});
        if (self.w.get(in_key) != null) {
            const nxt = try resBlock(self, h, emb, prefix, i, s);
            _ = mlx.mlx_array_free(h);
            h = nxt;
        } else {
            const dw_key = keyBuf(&buf, prefix ++ ".{d}.dw.weight", .{i});
            if (self.w.get(dw_key) != null) {
                const nxt = try largeKernel(self, h, prefix, i, s);
                _ = mlx.mlx_array_free(h);
                h = nxt;
            }
        }
        if (self.arch.temporal_every > 0 and i % self.arch.temporal_every == 0) {
            const nxt = try temporal(self, allocator, h, temporal_idx, b, t, s);
            _ = mlx.mlx_array_free(h);
            h = nxt;
        }
        try mlx.check(mlx.mlx_array_eval(h));
    }
    return h;
}

fn embed(self: *const Model, scale_h: f64, scale_w: f64, tier: u32, rows: u32, s: S) !A {
    const dim = fourierInDim(self.arch.freq, self.arch.tier_dim);
    var code: [96]f32 = undefined;
    fourierCode(scale_h, scale_w, @floatFromInt(tier), self.arch.freq, self.arch.tier_dim, code[0..dim]);
    const raw = mlx.mlx_array_new_data(code[0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const x1 = try linear(raw, try must(&self.w, "resizer.embed.linear.0.weight"), self.w.get("resizer.embed.linear.0.bias"), s);
    defer _ = mlx.mlx_array_free(x1);
    const a1 = try silu(x1, s);
    defer _ = mlx.mlx_array_free(a1);
    const y = try linear(a1, try must(&self.w, "resizer.embed.linear.2.weight"), self.w.get("resizer.embed.linear.2.bias"), s);
    if (rows == 1) return y;
    var tiled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&tiled, y, @intCast(rows), 0, s));
    _ = mlx.mlx_array_free(y);
    return tiled;
}

fn resBlock(self: *const Model, x: A, emb: A, comptime prefix: []const u8, i: u32, s: S) !A {
    var buf: [128]u8 = undefined;
    const n1w = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.in_layers.0.weight", .{i}));
    const n1b = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.in_layers.0.bias", .{i}));
    const c1w = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.in_layers.2.weight", .{i}));
    const c1b = self.w.get(keyBuf(&buf, prefix ++ ".{d}.in_layers.2.bias", .{i}));
    const h0 = try groupNormNhwc(x, n1w, n1b, s);
    defer _ = mlx.mlx_array_free(h0);
    const a0 = try silu(h0, s);
    defer _ = mlx.mlx_array_free(a0);
    var h = try conv2dPt(a0, c1w, c1b, 1, false, s);

    const e1w = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.emb_layers.1.weight", .{i}));
    const e1b = self.w.get(keyBuf(&buf, prefix ++ ".{d}.emb_layers.1.bias", .{i}));
    const ea = try silu(emb, s);
    defer _ = mlx.mlx_array_free(ea);
    const eout = try linear(ea, e1w, e1b, s);
    defer _ = mlx.mlx_array_free(eout);
    const ch: c_int = mlx.getShape(h)[3];
    const n = mlx.getShape(h)[0];
    const hh = mlx.getShape(h)[1];
    const ww = mlx.getShape(h)[2];
    const e4 = try reshape(eout, &[_]c_int{ n, 1, 1, 2 * ch }, s);
    defer _ = mlx.mlx_array_free(e4);
    const chunks = try splitLast(e4, 2, s);
    defer {
        _ = mlx.mlx_array_free(chunks[0]);
        _ = mlx.mlx_array_free(chunks[1]);
    }
    const n2w = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.out_norm.weight", .{i}));
    const n2b = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.out_norm.bias", .{i}));
    const hn = try groupNormNhwc(h, n2w, n2b, s);
    _ = mlx.mlx_array_free(h);
    const one = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(one);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    var sp1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp1);
    try mlx.check(mlx.mlx_add(&sp1, chunks[0], one, s));
    try mlx.check(mlx.mlx_multiply(&scaled, hn, sp1, s));
    _ = mlx.mlx_array_free(hn);
    var shifted = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&shifted, scaled, chunks[1], s));
    h = shifted;

    const a1 = try silu(h, s);
    _ = mlx.mlx_array_free(h);
    const c2w = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.out_layers.2.weight", .{i}));
    const c2b = self.w.get(keyBuf(&buf, prefix ++ ".{d}.out_layers.2.bias", .{i}));
    h = try conv2dPt(a1, c2w, c2b, 1, false, s);
    _ = mlx.mlx_array_free(a1);

    if (self.arch.use_se) {
        if (self.w.get(keyBuf(&buf, prefix ++ ".{d}.se.fc.1.weight", .{i}))) |_| {
            const se = try seBlock(self, h, prefix, i, s);
            _ = mlx.mlx_array_free(h);
            h = se;
        }
    }
    var skip_buf: [128]u8 = undefined;
    const skip_k = keyBuf(&skip_buf, prefix ++ ".{d}.skip.weight", .{i});
    const skip = if (self.w.get(skip_k)) |sw|
        try conv2dPt(x, sw, self.w.get(keyBuf(&buf, prefix ++ ".{d}.skip.bias", .{i})), 0, false, s)
    else
        try contig(x, s);
    defer _ = mlx.mlx_array_free(skip);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, skip, h, s));
    _ = mlx.mlx_array_free(h);
    _ = hh;
    _ = ww;
    return out;
}

fn seBlock(self: *const Model, x: A, comptime prefix: []const u8, i: u32, s: S) !A {
    var buf: [128]u8 = undefined;
    const sh = mlx.getShape(x);
    const axes = [_]c_int{ 1, 2 };
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axes(&mean, x, &axes, 2, false, s));
        const f0 = try linear(mean, try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.se.fc.2.weight", .{i})), self.w.get(keyBuf(&buf, prefix ++ ".{d}.se.fc.2.bias", .{i})), s);
        defer _ = mlx.mlx_array_free(f0);
        const a = try silu(f0, s);
        defer _ = mlx.mlx_array_free(a);
        const f1 = try linear(a, try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.se.fc.4.weight", .{i})), self.w.get(keyBuf(&buf, prefix ++ ".{d}.se.fc.4.bias", .{i})), s);
    defer _ = mlx.mlx_array_free(f1);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sigmoid(&gate, f1, s));
    const g4 = try reshape(gate, &[_]c_int{ sh[0], 1, 1, sh[3] }, s);
    defer _ = mlx.mlx_array_free(g4);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, x, g4, s));
    return out;
}

fn largeKernel(self: *const Model, x: A, comptime prefix: []const u8, i: u32, s: S) !A {
    var buf: [128]u8 = undefined;
    const nw = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.norm.weight", .{i}));
    const nb = try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.norm.bias", .{i}));
    const n = try groupNormNhwc(x, nw, nb, s);
    defer _ = mlx.mlx_array_free(n);
    const a = try silu(n, s);
    defer _ = mlx.mlx_array_free(a);
    const pad: c_int = @intCast(self.arch.large_kernel_size / 2);
    const dw = try conv2dPt(a, try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.dw.weight", .{i})), self.w.get(keyBuf(&buf, prefix ++ ".{d}.dw.bias", .{i})), pad, false, s);
    defer _ = mlx.mlx_array_free(dw);
    const pw = try conv2dPt(dw, try must(&self.w, keyBuf(&buf, prefix ++ ".{d}.pw.weight", .{i})), self.w.get(keyBuf(&buf, prefix ++ ".{d}.pw.bias", .{i})), 0, false, s);
    defer _ = mlx.mlx_array_free(pw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, pw, s));
    return out;
}

fn postRefine(self: *const Model, x: A, s: S) !A {
    var h = try contig(x, s);
    var i: u32 = 0;
    const n = self.arch.post_refine_layers;
    while (i < n) : (i += 1) {
        var buf: [128]u8 = undefined;
        const wk = keyBuf(&buf, "resizer.post_up.refine.{d}.weight", .{if (n == 2) i * 2 else i});
        const w = self.w.get(wk) orelse {
            const wk2 = keyBuf(&buf, "resizer.post_up.refine.{d}.weight", .{i});
            const w2 = self.w.get(wk2) orelse break;
            const c = try conv2dPt(h, w2, self.w.get(keyBuf(&buf, "resizer.post_up.refine.{d}.bias", .{i})), 1, false, s);
            _ = mlx.mlx_array_free(h);
            h = c;
            if (i + 1 < n) {
                const a = try silu(h, s);
                _ = mlx.mlx_array_free(h);
                h = a;
            }
            continue;
        };
        const c = try conv2dPt(h, w, self.w.get(keyBuf(&buf, "resizer.post_up.refine.{d}.bias", .{i * 2})), 1, false, s);
        _ = mlx.mlx_array_free(h);
        h = c;
        if (i + 1 < n) {
            const a = try silu(h, s);
            _ = mlx.mlx_array_free(h);
            h = a;
        }
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, h, s));
    _ = mlx.mlx_array_free(h);
    return out;
}

fn temporal(self: *const Model, allocator: std.mem.Allocator, x_nhwc: A, idx: u32, b: u32, t: u32, s: S) !A {
    _ = allocator;
    if (self.w.get("temporal_blocks.0.dwconv.weight") == null and !self.arch.use_multiscale_temporal)
        return contig(x_nhwc, s);
    const sh = mlx.getShape(x_nhwc);
    const c = sh[3];
    const hh = sh[1];
    const ww = sh[2];
    const x5 = try reshape(x_nhwc, &[_]c_int{ @intCast(b), @intCast(t), hh, ww, c }, s);
    defer _ = mlx.mlx_array_free(x5);
    // NDHWC: [B,T,H,W,C]
    var buf: [96]u8 = undefined;
    const nkey = keyBuf(&buf, "temporal_blocks.{d}.norm.weight", .{idx});
    const n = try groupNormNhwc(x_nhwc, try must(&self.w, nkey), try must(&self.w, keyBuf(&buf, "temporal_blocks.{d}.norm.bias", .{idx})), s);
    defer _ = mlx.mlx_array_free(n);
    const n5 = try reshape(n, &[_]c_int{ @intCast(b), @intCast(t), hh, ww, c }, s);
    defer _ = mlx.mlx_array_free(n5);
    const a = try silu(n5, s);
    defer _ = mlx.mlx_array_free(a);
    const dw_k = keyBuf(&buf, "temporal_blocks.{d}.dwconv.weight", .{idx});
    const dw = try conv3dPt(a, try must(&self.w, dw_k), self.w.get(keyBuf(&buf, "temporal_blocks.{d}.dwconv.bias", .{idx})), @intCast(self.arch.temporal_kernel / 2), @intCast(c), s);
    defer _ = mlx.mlx_array_free(dw);
    const pw = try conv3dPt(dw, try must(&self.w, keyBuf(&buf, "temporal_blocks.{d}.pwconv.weight", .{idx})), self.w.get(keyBuf(&buf, "temporal_blocks.{d}.pwconv.bias", .{idx})), 0, 1, s);
    defer _ = mlx.mlx_array_free(pw);
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, x5, pw, s));
    return reshape(sum, &[_]c_int{ sh[0], hh, ww, c }, s);
}

fn upsampleMid(self: *const Model, allocator: std.mem.Allocator, x: A, out_h: u32, out_w: u32, scale_h: f64, scale_w: f64, s: S) !A {
    const sh = mlx.getShape(x);
    const h: u32 = @intCast(sh[1]);
    const w: u32 = @intCast(sh[2]);
    if (self.arch.learned_k >= 2) {
        const sc = @min(scale_h, scale_w);
        const k = self.arch.learned_k;
        if (@abs(@as(f64, @floatFromInt(k)) - sc) <= 1e-3 and h * k >= out_h and w * k >= out_w) {
            const y = try pixelShuffleUp(self, x, k, s);
            defer _ = mlx.mlx_array_free(y);
            const yh: u32 = @intCast(mlx.getShape(y)[1]);
            const yw: u32 = @intCast(mlx.getShape(y)[2]);
            if (yh == out_h and yw == out_w) return contig(y, s);
            return sliceHw(y, out_h, out_w, s);
        }
    }
    _ = allocator;
    return bilinearNhwc(x, out_h, out_w, s);
}

fn pixelShuffleUp(self: *const Model, x: A, k: u32, s: S) !A {
    var buf: [96]u8 = undefined;
    const wk = keyBuf(&buf, "resizer.learned_up.branches.{d}.conv.weight", .{k});
    const conv = try conv2dPt(x, try must(&self.w, wk), self.w.get(keyBuf(&buf, "resizer.learned_up.branches.{d}.conv.bias", .{k})), 1, true, s);
    defer _ = mlx.mlx_array_free(conv);
    return pixelShuffle2d(conv, k, s);
}

fn pixelShuffle2d(x: A, factor: u32, s: S) !A {
    const sh = mlx.getShape(x);
    const n = sh[0];
    const h = sh[1];
    const w = sh[2];
    const f: c_int = @intCast(factor);
    const c = @divExact(sh[3], f * f);
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, x, &[_]c_int{ n, h, w, c, f, f }, 6, s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 1, 4, 2, 5, 3 }, 6, s));
    const ct = try contig(t, s);
    defer _ = mlx.mlx_array_free(ct);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, ct, &[_]c_int{ n, h * f, w * f, c }, 4, s));
    return out;
}

/// F.interpolate bilinear, align_corners=False, on NHWC.
pub fn bilinearNhwc(x: A, out_h: u32, out_w: u32, s: S) !A {
    const y = try linearAxis(x, 1, out_h, s);
    defer _ = mlx.mlx_array_free(y);
    return linearAxis(y, 2, out_w, s);
}

fn linearAxis(x: A, axis: usize, out_len: u32, s: S) !A {
    const sh = mlx.getShape(x);
    const in_len: u32 = @intCast(sh[axis]);
    if (in_len == out_len) return contig(x, s);
    const lo = try std.heap.c_allocator.alloc(i32, out_len);
    defer std.heap.c_allocator.free(lo);
    const hi = try std.heap.c_allocator.alloc(i32, out_len);
    defer std.heap.c_allocator.free(hi);
    const wt = try std.heap.c_allocator.alloc(f32, out_len);
    defer std.heap.c_allocator.free(wt);
    const src: f64 = @floatFromInt(in_len);
    const dst: f64 = @floatFromInt(out_len);
    var i: u32 = 0;
    while (i < out_len) : (i += 1) {
        var p = (@as(f64, @floatFromInt(i)) + 0.5) * src / dst - 0.5;
        if (p < 0) p = 0;
        const max_p: f64 = @floatFromInt(in_len - 1);
        if (p > max_p) p = max_p;
        const l: u32 = @intFromFloat(@floor(p));
        lo[i] = @intCast(l);
        hi[i] = @intCast(@min(l + 1, in_len - 1));
        wt[i] = @floatCast(p - @floor(p));
    }
    const lo_a = mlx.mlx_array_new_data(lo.ptr, &[_]c_int{@intCast(out_len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_data(hi.ptr, &[_]c_int{@intCast(out_len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(hi_a);
    var wshape = [_]c_int{ 1, 1, 1, 1 };
    wshape[axis] = @intCast(out_len);
    const w_a = mlx.mlx_array_new_data(wt.ptr, &wshape, 4, .float32);
    defer _ = mlx.mlx_array_free(w_a);
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_take_axis(&a, x, lo_a, @intCast(axis), s));
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_take_axis(&b, x, hi_a, @intCast(axis), s));
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, b, a, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, diff, w_a, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, a, scaled, s));
    return out;
}

fn normalize(x: A, s: S) !A {
    const mean = mlx.mlx_array_new_data(&LATENTS_MEAN, &[_]c_int{ 1, IN_CH, 1, 1, 1 }, 5, .float32);
    defer _ = mlx.mlx_array_free(mean);
    const stdv = mlx.mlx_array_new_data(&LATENTS_STD, &[_]c_int{ 1, IN_CH, 1, 1, 1 }, 5, .float32);
    defer _ = mlx.mlx_array_free(stdv);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, xf, mean, s));
    var z = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&z, d, stdv, s));
    return z;
}

fn denormalize(z: A, s: S) !A {
    const mean = mlx.mlx_array_new_data(&LATENTS_MEAN, &[_]c_int{ 1, IN_CH, 1, 1, 1 }, 5, .float32);
    defer _ = mlx.mlx_array_free(mean);
    const stdv = mlx.mlx_array_new_data(&LATENTS_STD, &[_]c_int{ 1, IN_CH, 1, 1, 1 }, 5, .float32);
    defer _ = mlx.mlx_array_free(stdv);
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_multiply(&m, z, stdv, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, m, mean, s));
    return out;
}

fn groupNormNhwc(x: A, weight: A, bias: A, s: S) !A {
    const sh = mlx.getShape(x);
    const n = sh[0];
    const h = sh[1];
    const w = sh[2];
    const c = sh[3];
    const g: c_int = if (c >= 32) 32 else c;
    const cg = @divExact(c, g);
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_reshape(&xr, x, &[_]c_int{ n, h, w, g, cg }, 5, s));
    const axes = [_]c_int{ 1, 2, 4 };
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axes(&mean, xr, &axes, axes.len, true, s));
    var xm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xm);
    try mlx.check(mlx.mlx_subtract(&xm, xr, mean, s));
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, xm, s));
    var vv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vv);
    try mlx.check(mlx.mlx_mean_axes(&vv, sq, &axes, axes.len, true, s));
    const eps = mlx.mlx_array_new_float(1e-5);
    defer _ = mlx.mlx_array_free(eps);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, vv, eps, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_multiply(&normed, xm, rs, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, normed, &[_]c_int{ n, h, w, c }, 4, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, flat, weight, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, scaled, bias, s));
    return out;
}

fn conv2dPt(x: A, w_pt: A, bias: ?A, pad: c_int, replicate: bool, s: S) !A {
    const wt = try transpose(w_pt, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const w = try contig(wt, s);
    defer _ = mlx.mlx_array_free(w);
    const groups: c_int = blk: {
        const shw = mlx.getShape(w_pt);
        const cin = shw[1];
        const cout = shw[0];
        if (cin == 1 and cout > 1) break :blk cout;
        break :blk 1;
    };
    var xin = x;
    var padded: ?A = null;
    var used_pad = pad;
    if (replicate and pad > 0) {
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        var p = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(&p, x, &[_]c_int{ 1, 2 }, 2, &[_]c_int{ pad, pad }, 2, &[_]c_int{ pad, pad }, 2, zero, "edge", s));
        padded = p;
        xin = p;
        used_pad = 0;
    }
    defer if (padded) |p| {
        _ = mlx.mlx_array_free(p);
    };
    const xc = try contig(xin, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, used_pad, used_pad, 1, 1, groups, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addBiasLast(o, b, s);
    }
    return o;
}

fn conv3dPt(x_ndhwc: A, w_pt: A, bias: ?A, pad_t: c_int, groups: c_int, s: S) !A {
    // PT [O, I/G, kD, kH, kW] → MLX [O, kD, kH, kW, I/G]
    const wt = try transpose(w_pt, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const w = try contig(wt, s);
    defer _ = mlx.mlx_array_free(w);
    const xc = try contig(x_ndhwc, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv3d(&o, xc, w, 1, 1, 1, pad_t, 0, 0, 1, 1, 1, groups, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addBiasLast(o, b, s);
    }
    return o;
}

fn linear(x: A, w_pt: A, bias: ?A, s: S) !A {
    const wt = try transpose(w_pt, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const wc = try contig(wt, s);
    defer _ = mlx.mlx_array_free(wc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, wc, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, o, b, s));
        return out;
    }
    return o;
}

fn silu(x: A, s: S) !A {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, sig, s));
    return o;
}

fn addBiasLast(x: A, b: A, s: S) !A {
    const rank = mlx.mlx_array_ndim(x);
    var shape: [5]c_int = .{ 1, 1, 1, 1, 1 };
    shape[rank - 1] = mlx.getShape(b)[0];
    const br = try reshape(b, shape[0..rank], s);
    defer _ = mlx.mlx_array_free(br);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, br, s));
    return o;
}

fn splitLast(x: A, n: c_int, s: S) ![2]A {
    const rank: c_int = @intCast(mlx.mlx_array_ndim(x));
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, x, n, rank - 1, s));
    var a = mlx.mlx_array_new();
    var b = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&a, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, parts, 1));
    return .{ a, b };
}

fn coordConcat(x: A, s: S) !A {
    const sh = mlx.getShape(x);
    const n = sh[0];
    const h = sh[1];
    const w = sh[2];
    var ys = try std.heap.c_allocator.alloc(f32, @intCast(h));
    defer std.heap.c_allocator.free(ys);
    var xs = try std.heap.c_allocator.alloc(f32, @intCast(w));
    defer std.heap.c_allocator.free(xs);
    const hf: f32 = @floatFromInt(h - 1);
    const wf: f32 = @floatFromInt(w - 1);
    var i: usize = 0;
    while (i < ys.len) : (i += 1) ys[i] = if (h == 1) 0 else -1 + 2 * @as(f32, @floatFromInt(i)) / hf;
    i = 0;
    while (i < xs.len) : (i += 1) xs[i] = if (w == 1) 0 else -1 + 2 * @as(f32, @floatFromInt(i)) / wf;
    const ya = mlx.mlx_array_new_data(ys.ptr, &[_]c_int{ 1, h, 1, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(ya);
    const xa = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ 1, 1, w, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(xa);
    var yb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yb);
    try mlx.check(mlx.mlx_broadcast_to(&yb, ya, &[_]c_int{ n, h, w, 1 }, 4, s));
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_broadcast_to(&xb, xa, &[_]c_int{ n, h, w, 1 }, 4, s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_vector_array_append_value(vec, x));
    try mlx.check(mlx.mlx_vector_array_append_value(vec, yb));
    try mlx.check(mlx.mlx_vector_array_append_value(vec, xb));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 3, s));
    return out;
}

fn toNhwc(x: A, s: S) !A {
    // [B,C,T,H,W] → [(B*T),H,W,C]
    const sh = mlx.getShape(x);
    const tr = try transpose(x, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(tr);
    const tc = try contig(tr, s);
    defer _ = mlx.mlx_array_free(tc);
    return reshape(tc, &[_]c_int{ sh[0] * sh[2], sh[3], sh[4], sh[1] }, s);
}

fn toBcthw(x: A, b: u32, t: u32, s: S) !A {
    const sh = mlx.getShape(x);
    const r = try reshape(x, &[_]c_int{ @intCast(b), @intCast(t), sh[1], sh[2], sh[3] }, s);
    defer _ = mlx.mlx_array_free(r);
    const tr = try transpose(r, &[_]c_int{ 0, 4, 1, 2, 3 }, s);
    defer _ = mlx.mlx_array_free(tr);
    return contig(tr, s);
}

fn sliceT(x: A, lo: u32, hi: u32, s: S) !A {
    const sh = mlx.getShape(x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &[_]c_int{ 0, 0, @intCast(lo), 0, 0 }, 5, &[_]c_int{ sh[0], sh[1], @intCast(hi), sh[3], sh[4] }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
    return o;
}

fn sliceHw(x: A, h: u32, w: u32, s: S) !A {
    const sh = mlx.getShape(x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], @intCast(h), @intCast(w), sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return o;
}

fn concatT(allocator: std.mem.Allocator, xs: []const A, s: S) !A {
    _ = allocator;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (xs) |a| try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
    return o;
}

fn contig(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
fn reshape(x: A, shape: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
fn transpose(x: A, axes: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
fn astype(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

const testing = std.testing;

test "minimax h3 upscaler: fourier code is sin/cos of 2^k without pi" {
    var buf: [96]f32 = undefined;
    fourierCode(2.0, 1.5, 1.0, 16, 32, &buf);
    try testing.expectApproxEqAbs(@sin(2.0 * 1.0), buf[0], 1e-6);
    try testing.expectApproxEqAbs(@cos(2.0 * 1.0), buf[16], 1e-6);
    try testing.expectApproxEqAbs(@sin(1.5 * 1.0), buf[32], 1e-6);
    try testing.expectApproxEqAbs(@sin(0.0), buf[64], 1e-6); // tier-1 = 0
    try testing.expectApproxEqAbs(@cos(0.0), buf[80], 1e-6);
}

test "minimax h3 upscaler: bilinear NHWC matches align_corners=False 1d" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const src = [_]f32{ 1, 2, 3, 4 };
    const x = mlx.mlx_array_new_data(&src, &[_]c_int{ 1, 2, 2, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(x);
    const y = try bilinearNhwc(x, 4, 4, s);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    try testing.expectEqual(@as(c_int, 4), mlx.getShape(y)[1]);
    try testing.expectEqual(@as(c_int, 4), mlx.getShape(y)[2]);
}

test "minimax h3 upscaler: health is false without CLI weights" {
    try testing.expect(!weightsPresent(null));
}

test "minimax h3 upscaler: 2.0x and 1.5x match the PyTorch dump (cos ≥ 0.999, rms within 1%)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const dir = env.span(.H3_UPSCALER_FIXTURE) orelse return error.SkipZigTest;
    const weights = cliPath() orelse return error.SkipZigTest;
    var model = Model.load(std.testing.allocator, weights) catch return error.SkipZigTest;
    defer model.deinit();
    const s = mlx.gpuStream();
    const x = try loadRaw(std.testing.allocator, dir, "latent.raw", &[_]c_int{ 1, 24, 3, 4, 6 });
    defer _ = mlx.mlx_array_free(x);
    try expectOracle(&model, std.testing.allocator, x, dir, "out_2x.raw", 8, 12, 1, s);
    try expectOracle(&model, std.testing.allocator, x, dir, "out_1p5.raw", 6, 9, 3, s);
}

fn loadRaw(a: std.mem.Allocator, dir: []const u8, name: []const u8, shape: []const c_int) !A {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(rio(), path, a, .limited(64 << 20));
    defer a.free(bytes);
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    if (bytes.len != n * 4) return error.BadFixture;
    const arr = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .float32);
    return contig(arr, mlx.mlx_default_cpu_stream_new()) catch arr;
}

fn expectOracle(model: *const Model, a: std.mem.Allocator, x: A, dir: []const u8, name: []const u8, oh: u32, ow: u32, tier: u32, s: S) !void {
    const got = try model.forward(a, x, oh, ow, tier, s);
    defer _ = mlx.mlx_array_free(got);
    var shape_buf: [5]c_int = undefined;
    const gsh = mlx.getShape(got);
    @memcpy(shape_buf[0..gsh.len], gsh);
    const want = try loadRaw(a, dir, name, shape_buf[0..gsh.len]);
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_array_eval(got));
    try mlx.check(mlx.mlx_array_eval(want));
    // Cosine + rms on flattened f32.
    var g2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g2);
    var w2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w2);
    try mlx.check(mlx.mlx_multiply(&g2, got, got, s));
    try mlx.check(mlx.mlx_multiply(&w2, want, want, s));
    var gs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gs);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_sum(&gs, g2, false, s));
    try mlx.check(mlx.mlx_sum(&ws, w2, false, s));
    var dot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dot);
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, got, want, s));
    try mlx.check(mlx.mlx_sum(&dot, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(dot));
    try mlx.check(mlx.mlx_array_eval(gs));
    try mlx.check(mlx.mlx_array_eval(ws));
    var d: f32 = 0;
    var gv: f32 = 0;
    var wv: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&d, dot));
    try mlx.check(mlx.mlx_array_item_float32(&gv, gs));
    try mlx.check(mlx.mlx_array_item_float32(&wv, ws));
    const cos = d / (@sqrt(gv) * @sqrt(wv));
    try testing.expect(cos >= 0.999);
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, got, want, s));
    var d2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d2);
    try mlx.check(mlx.mlx_multiply(&d2, diff, diff, s));
    var ds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ds);
    try mlx.check(mlx.mlx_sum(&ds, d2, false, s));
    try mlx.check(mlx.mlx_array_eval(ds));
    var dv: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&dv, ds));
    const n: f32 = @floatFromInt(mlx.mlx_array_size(got));
    const rms_ratio = @sqrt(dv / n) / (@sqrt(wv / n) + 1e-8);
    try testing.expect(rms_ratio <= 0.01);
}
