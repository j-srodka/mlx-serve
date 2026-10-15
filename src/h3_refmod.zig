//! MiniMax-H3 RefMod: VAE latents saved as Comfy-compatible v4 safetensors.
//! Visual tensor `latent` is `[1, 24, T, H, W]` f16; voice is `[1, 32, 2, T]`.
//! `__metadata__.refmod_meta` is JSON.

const std = @import("std");
const mlx = @import("mlx.zig");
const minimax_h3 = @import("minimax_h3.zig");
const minimax_h3_audio = @import("minimax_h3_audio.zig");
const testing = std.testing;

pub const LATENT_C: u32 = 24;
pub const AUDIO_LATENT_C: u32 = minimax_h3_audio.VAE_LATENT_CHANNELS;
pub const AUDIO_STEREO: u32 = 2;
pub const META_KEY: []const u8 = "refmod_meta";
pub const TENSOR_KEY: []const u8 = "latent";

pub const Meta = struct {
    kind: enum { image, video, audio },
    latent_t: u32,
    latent_h: u32,
    latent_w: u32,
    canvas_w: u32,
    canvas_h: u32,
    mode: []const u8 = "encode",
    /// Source frames when `kind == .video` (stacked stills along T).
    source_frames: u32 = 0,

    pub fn condVisualTokens(self: Meta) u32 {
        if (self.kind == .audio) return 0;
        const rows = (self.latent_h / minimax_h3.PATCH_H) * (self.latent_w / minimax_h3.PATCH_W);
        return switch (self.kind) {
            .image => rows,
            .video => self.latent_t * rows,
            .audio => 0,
        };
    }

    pub fn toResolvedRef(self: Meta, ordinal: u32, src_index: usize) minimax_h3.ResolvedRef {
        const canvas = minimax_h3.Canvas{ .w = self.canvas_w, .h = self.canvas_h };
        return switch (self.kind) {
            .audio => .{
                .kind = .audio,
                .ordinal = ordinal,
                .src_index = src_index,
                .audio_t = self.latent_t,
            },
            .image => .{
                .kind = .image,
                .ordinal = ordinal,
                .src_index = src_index,
                .canvas = canvas,
                .latent_h = self.latent_h,
                .latent_w = self.latent_w,
            },
            .video => .{
                .kind = .video,
                .ordinal = ordinal,
                .src_index = src_index,
                .canvas = canvas,
                .frames = if (self.source_frames != 0) self.source_frames else self.latent_t,
                .latent_t = self.latent_t,
                .latent_h = self.latent_h,
                .latent_w = self.latent_w,
            },
        };
    }
};

pub const Loaded = struct {
    meta: Meta,
    /// Visual `[1, 24, T, H, W]` or audio `[1, 32, 2, T]`.
    latent: mlx.mlx_array,

    pub fn deinit(self: *Loaded) void {
        _ = mlx.mlx_array_free(self.latent);
        self.* = undefined;
    }

    pub fn takeLatent(self: *Loaded) mlx.mlx_array {
        const lat = self.latent;
        self.latent = undefined;
        return lat;
    }
};

pub const LoadError = error{
    BadPath,
    MissingTensor,
    BadTensorRank,
    BadTensorShape,
    MissingMeta,
    BadMetaJson,
    OutOfMemory,
    MlxError,
};

fn parseMetaJson(allocator: std.mem.Allocator, raw: []const u8) !Meta {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;

    const kind_str = obj.get("kind") orelse return error.BadMetaJson;
    if (kind_str != .string) return error.BadMetaJson;
    const kind: @TypeOf(@as(Meta, undefined).kind) = if (std.mem.eql(u8, kind_str.string, "video"))
        .video
    else if (std.mem.eql(u8, kind_str.string, "image"))
        .image
    else if (std.mem.eql(u8, kind_str.string, "audio"))
        .audio
    else
        return error.BadMetaJson;

    const lt = jsonU32(obj, "latent_t") orelse return error.BadMetaJson;
    const lh = jsonU32(obj, "latent_h") orelse if (kind == .audio) 0 else return error.BadMetaJson;
    const lw = jsonU32(obj, "latent_w") orelse if (kind == .audio) 0 else return error.BadMetaJson;
    // Community Comfy packs (e.g. malcolmrey) omit canvas_*; infer from VAE /16.
    const cw = jsonU32(obj, "canvas_w") orelse if (kind == .audio) 0 else lw * minimax_h3.VAE_SPATIAL;
    const ch = jsonU32(obj, "canvas_h") orelse if (kind == .audio) 0 else lh * minimax_h3.VAE_SPATIAL;
    const mode = if (obj.get("mode")) |m| switch (m) {
        .string => m.string,
        else => "encode",
    } else "encode";
    const sf = jsonU32(obj, "source_frames") orelse 0;

    if (lt == 0) return error.BadMetaJson;
    if (kind != .audio and (lh == 0 or lw == 0 or cw == 0 or ch == 0)) return error.BadMetaJson;
    return .{
        .kind = kind,
        .latent_t = lt,
        .latent_h = lh,
        .latent_w = lw,
        .canvas_w = cw,
        .canvas_h = ch,
        .mode = mode,
        .source_frames = sf,
    };
}

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (f >= 0) @intFromFloat(f) else null,
        else => null,
    };
}

pub fn formatMetaJson(allocator: std.mem.Allocator, meta: Meta) ![]u8 {
    return std.fmt.allocPrint(allocator, "{{\"kind\":\"{s}\",\"latent_t\":{d},\"latent_h\":{d},\"latent_w\":{d},\"canvas_w\":{d},\"canvas_h\":{d},\"mode\":\"{s}\",\"source_frames\":{d}}}", .{
        @tagName(meta.kind),
        meta.latent_t,
        meta.latent_h,
        meta.latent_w,
        meta.canvas_w,
        meta.canvas_h,
        meta.mode,
        meta.source_frames,
    });
}

pub fn validateAbsoluteSafetensorsPath(io: std.Io, path: []const u8) LoadError!void {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadPath;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.BadPath;
    f.close(io);
}

/// Create-path twin of `validateAbsoluteSafetensorsPath`: an OUTPUT file need not
/// exist yet, so the path's shape and its parent directory are checked instead.
/// A save that fails later reaches genJobRun, which only logs and never answers.
pub fn validateSafetensorsOutPath(io: std.Io, path: []const u8) LoadError!void {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadPath;
    if (!std.mem.endsWith(u8, path, ".safetensors")) return error.BadPath;
    const dir = std.fs.path.dirname(path) orelse return error.BadPath;
    const d = std.Io.Dir.openDirAbsolute(io, dir, .{}) catch return error.BadPath;
    d.close(io);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, _: mlx.mlx_stream) LoadError!Loaded {
    try validateAbsoluteSafetensorsPath(io, path);

    const pathz = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(pathz);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, pathz, cpu));

    var latent_raw = mlx.mlx_array_new();
    const tkey = try allocator.dupeSentinel(u8, TENSOR_KEY, 0);
    defer allocator.free(tkey);
    if (mlx.mlx_map_string_to_array_get(&latent_raw, tensor_map, tkey) != 0) return error.MissingTensor;

    var meta_raw: [*:0]const u8 = undefined;
    const mkey = try allocator.dupeSentinel(u8, META_KEY, 0);
    defer allocator.free(mkey);
    if (mlx.mlx_map_string_to_string_get(&meta_raw, meta_map, mkey) != 0) {
        _ = mlx.mlx_array_free(latent_raw);
        return error.MissingMeta;
    }
    const meta = parseMetaJson(allocator, std.mem.sliceTo(meta_raw, 0)) catch {
        _ = mlx.mlx_array_free(latent_raw);
        return error.BadMetaJson;
    };

    const shp = mlx.getShape(latent_raw);
    switch (meta.kind) {
        .audio => {
            if (shp.len != 4 or shp[0] != 1 or shp[1] != AUDIO_LATENT_C or shp[2] != AUDIO_STEREO) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorShape;
            }
            const t: u32 = @intCast(shp[3]);
            if (t != meta.latent_t) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorShape;
            }
        },
        .image, .video => {
            if (shp.len != 5) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorRank;
            }
            if (shp[0] != 1 or shp[1] != LATENT_C) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorShape;
            }
            const t: u32 = @intCast(shp[2]);
            const h: u32 = @intCast(shp[3]);
            const w: u32 = @intCast(shp[4]);
            if (t != meta.latent_t or h != meta.latent_h or w != meta.latent_w) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorShape;
            }
            if (meta.kind == .image and t != 1) {
                _ = mlx.mlx_array_free(latent_raw);
                return error.BadTensorShape;
            }
        },
    }

    _ = mlx.mlx_array_eval(latent_raw);
    return .{ .meta = meta, .latent = latent_raw };
}

pub fn save(allocator: std.mem.Allocator, path: []const u8, meta: Meta, latent: mlx.mlx_array) !void {
    const shp = mlx.getShape(latent);
    switch (meta.kind) {
        .audio => {
            if (shp.len != 4 or shp[0] != 1 or shp[1] != AUDIO_LATENT_C or shp[2] != AUDIO_STEREO) return error.BadTensorShape;
            if (@as(u32, @intCast(shp[3])) != meta.latent_t) return error.BadTensorShape;
        },
        .image, .video => {
            if (shp.len != 5 or shp[0] != 1 or shp[1] != LATENT_C) return error.BadTensorShape;
            if (@as(u32, @intCast(shp[2])) != meta.latent_t) return error.BadTensorShape;
        },
    }

    const pathz = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(pathz);

    const meta_json = try formatMetaJson(allocator, meta);
    defer allocator.free(meta_json);
    const meta_z = try allocator.dupeSentinel(u8, meta_json, 0);
    defer allocator.free(meta_z);

    const tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    const meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    const tkey = try allocator.dupeSentinel(u8, TENSOR_KEY, 0);
    defer allocator.free(tkey);
    const mkey = try allocator.dupeSentinel(u8, META_KEY, 0);
    defer allocator.free(mkey);

    try mlx.check(mlx.mlx_map_string_to_array_insert(tensor_map, tkey, latent));
    try mlx.check(mlx.mlx_map_string_to_string_insert(meta_map, mkey, meta_z));

    try mlx.check(mlx.mlx_save_safetensors(pathz, tensor_map, meta_map));
}

/// Concatenates `parts` along the latent T axis. The result is ALWAYS a fresh
/// owned handle — a single part is copied, never returned by alias, so a caller
/// may free the result and its inputs independently.
pub fn stackLatentsAlongT(_: std.mem.Allocator, parts: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (parts.len == 0) return error.BadTensorShape;

    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

pub fn metaFromAudioLatent(latent: mlx.mlx_array) !Meta {
    const shp = mlx.getShape(latent);
    if (shp.len != 4 or shp[0] != 1 or shp[1] != AUDIO_LATENT_C or shp[2] != AUDIO_STEREO) return error.BadTensorShape;
    const t: u32 = @intCast(shp[3]);
    if (t == 0) return error.BadTensorShape;
    return .{
        .kind = .audio,
        .latent_t = t,
        .latent_h = 0,
        .latent_w = 0,
        .canvas_w = 0,
        .canvas_h = 0,
        .mode = "encode",
        .source_frames = 0,
    };
}

pub fn metaFromLatentShape(canvas_w: u32, canvas_h: u32, latent: mlx.mlx_array) !Meta {
    const shp = mlx.getShape(latent);
    if (shp.len != 5 or shp[0] != 1 or shp[1] != LATENT_C) return error.BadTensorShape;
    const t: u32 = @intCast(shp[2]);
    const lh: u32 = @intCast(shp[3]);
    const lw: u32 = @intCast(shp[4]);
    const kind: @TypeOf(@as(Meta, undefined).kind) = if (t > 1) .video else .image;
    return .{
        .kind = kind,
        .latent_t = t,
        .latent_h = lh,
        .latent_w = lw,
        .canvas_w = canvas_w,
        .canvas_h = canvas_h,
        .mode = "encode",
        .source_frames = if (kind == .video) t else 0,
    };
}

// ── Tests (no GPU) ──────────────────────────────────────────────────────────

test "refmod meta json roundtrip and cond tokens" {
    const a = testing.allocator;
    const m: Meta = .{
        .kind = .image,
        .latent_t = 1,
        .latent_h = 26,
        .latent_w = 48,
        .canvas_w = 768,
        .canvas_h = 416,
    };
    const j = try formatMetaJson(a, m);
    defer a.free(j);
    const back = try parseMetaJson(a, j);
    try testing.expect(back.kind == .image);
    try testing.expectEqual(@as(u32, 26), back.latent_h);
    try testing.expectEqual(@as(u32, 312), m.condVisualTokens());
}

test "refmod meta infers canvas when community pack omits it" {
    const a = testing.allocator;
    const j =
        "{\"kind\":\"video\",\"latent_t\":22,\"latent_h\":32,\"latent_w\":48,\"mode\":\"encode\",\"name\":\"x\",\"_format_version\":2}";
    const back = try parseMetaJson(a, j);
    try testing.expect(back.kind == .video);
    try testing.expectEqual(@as(u32, 22), back.latent_t);
    try testing.expectEqual(@as(u32, 768), back.canvas_w);
    try testing.expectEqual(@as(u32, 512), back.canvas_h);
}

test "refmod load community aaliyah pack" {
    const a = testing.allocator;
    const path = "/tmp/malcolmrey-h3/minimaxh3_aaliyah_v1_refmod.safetensors";
    const io = std.Io.Threaded.global_single_threaded.io();
    const probe = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.SkipZigTest;
    probe.close(io);
    const s = mlx.gpuStream();
    var loaded = try load(a, io, path, s);
    defer loaded.deinit();
    try testing.expect(loaded.meta.kind == .video);
    try testing.expectEqual(@as(u32, 22), loaded.meta.latent_t);
    try testing.expectEqual(@as(u32, 32), loaded.meta.latent_h);
    try testing.expectEqual(@as(u32, 32), loaded.meta.latent_w);
    try testing.expectEqual(@as(u32, 512), loaded.meta.canvas_w);
    try testing.expectEqual(@as(u32, 512), loaded.meta.canvas_h);
}

test "refmod save and load tiny latent" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrint(a, "{s}/pack.safetensors", .{root_buf[0..root_len]});
    defer a.free(path);

    const t: u32 = 1;
    const lh: u32 = 4;
    const lw: u32 = 4;
    const n = 1 * LATENT_C * t * lh * lw;
    const buf = try a.alloc(f16, n);
    defer a.free(buf);
    @memset(buf, @as(f16, 0.25));
    const shp = [_]c_int{ 1, @intCast(LATENT_C), @intCast(t), @intCast(lh), @intCast(lw) };
    const lat = mlx.mlx_array_new_data(buf.ptr, &shp, 5, .float16);
    defer _ = mlx.mlx_array_free(lat);

    const meta: Meta = .{
        .kind = .image,
        .latent_t = t,
        .latent_h = lh,
        .latent_w = lw,
        .canvas_w = 64,
        .canvas_h = 64,
    };
    try save(a, path, meta, lat);

    const s = mlx.gpuStream();
    var loaded = try load(a, io, path, s);
    defer loaded.deinit();
    try testing.expect(loaded.meta.kind == .image);
    try testing.expectEqual(@as(u32, 4), loaded.meta.latent_h);
}

test "refmod load rejects missing tensor in header-only fixture" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrint(a, "{s}/empty.safetensors", .{root_buf[0..root_len]});
    defer a.free(path);

    // Valid v4 safetensors with an empty tensor map (no `latent` key). A raw JSON
    // blob is not a safetensors file and can abort inside mlx_load_safetensors.
    const header_json = "{}";
    var file_buf: [8 + header_json.len]u8 = undefined;
    std.mem.writeInt(u64, file_buf[0..8], header_json.len, .little);
    @memcpy(file_buf[8..], header_json);
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.safetensors", .data = &file_buf });

    const s = mlx.gpuStream();
    try testing.expectError(error.MissingTensor, load(a, io, path, s));
}

test "refmod single-part stack owns its handle instead of aliasing the input" {
    const a = testing.allocator;
    const t: u32 = 1;
    const lh: u32 = 4;
    const lw: u32 = 4;
    const n = 1 * LATENT_C * t * lh * lw;
    const buf = try a.alloc(f16, n);
    defer a.free(buf);
    @memset(buf, @as(f16, 0.5));
    const shp = [_]c_int{ 1, @intCast(LATENT_C), @intCast(t), @intCast(lh), @intCast(lw) };
    const one = mlx.mlx_array_new_data(buf.ptr, &shp, 5, .float16);
    defer _ = mlx.mlx_array_free(one);

    const s = mlx.gpuStream();
    const parts = [_]mlx.mlx_array{one};
    const stacked = try stackLatentsAlongT(a, &parts, s);
    defer _ = mlx.mlx_array_free(stacked);

    // Both this test and the HTTP handler free the stack AND every input, so an
    // aliased handle would be a double `delete` on the same ctx.
    try testing.expect(stacked.ctx != one.ctx);
    try testing.expectEqual(@as(c_int, @intCast(t)), mlx.getShape(stacked)[2]);
}

test "refmod audio meta roundtrip and rank-4 save load" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrint(a, "{s}/voice.safetensors", .{root_buf[0..root_len]});
    defer a.free(path);

    const t: u32 = 3;
    const n = 1 * AUDIO_LATENT_C * AUDIO_STEREO * t;
    const buf = try a.alloc(f16, n);
    defer a.free(buf);
    @memset(buf, @as(f16, 0.125));
    const shp = [_]c_int{ 1, @intCast(AUDIO_LATENT_C), @intCast(AUDIO_STEREO), @intCast(t) };
    const lat = mlx.mlx_array_new_data(buf.ptr, &shp, 4, .float16);
    defer _ = mlx.mlx_array_free(lat);

    const meta = try metaFromAudioLatent(lat);
    try save(a, path, meta, lat);

    const s = mlx.gpuStream();
    var loaded = try load(a, io, path, s);
    defer loaded.deinit();
    try testing.expect(loaded.meta.kind == .audio);
    try testing.expectEqual(t, loaded.meta.latent_t);
    const lshp = mlx.getShape(loaded.latent);
    try testing.expectEqual(@as(c_int, 4), @as(c_int, @intCast(lshp.len)));
}

test "refmod audio meta resolves to audio ref with audio_t" {
    const meta: Meta = .{
        .kind = .audio,
        .latent_t = 20,
        .latent_h = 0,
        .latent_w = 0,
        .canvas_w = 0,
        .canvas_h = 0,
    };
    const r = meta.toResolvedRef(1, 0);
    try testing.expect(r.kind == .audio);
    try testing.expectEqual(@as(u32, 1), r.ordinal);
    try testing.expectEqual(@as(u32, 20), r.audio_t);
    try testing.expectEqual(@as(usize, 0), r.src_index);
}

test "refmod save rejects visual meta with audio-shaped latent" {
    const a = testing.allocator;
    const t: u32 = 1;
    const n = 1 * AUDIO_LATENT_C * AUDIO_STEREO * t;
    const buf = try a.alloc(f16, n);
    defer a.free(buf);
    const shp = [_]c_int{ 1, @intCast(AUDIO_LATENT_C), @intCast(AUDIO_STEREO), @intCast(t) };
    const lat = mlx.mlx_array_new_data(buf.ptr, &shp, 4, .float16);
    defer _ = mlx.mlx_array_free(lat);
    const meta: Meta = .{
        .kind = .image,
        .latent_t = 1,
        .latent_h = 4,
        .latent_w = 4,
        .canvas_w = 64,
        .canvas_h = 64,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrint(a, "{s}/bad.safetensors", .{root_buf[0..root_len]});
    defer a.free(path);
    try testing.expectError(error.BadTensorShape, save(a, path, meta, lat));
}

test "refmod out_path validator accepts a file that does not exist yet" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrint(a, "{s}/pack.safetensors", .{root_buf[0..root_len]});
    defer a.free(path);

    try validateSafetensorsOutPath(io, path);
    try testing.expectError(error.BadPath, validateSafetensorsOutPath(io, "relative/pack.safetensors"));
    try testing.expectError(error.BadPath, validateSafetensorsOutPath(io, "/tmp/pack.bin"));
    try testing.expectError(error.BadPath, validateSafetensorsOutPath(io, ""));
    try testing.expectError(error.BadPath, validateSafetensorsOutPath(io, "/tmp/mlx-serve-no-such-dir-xyz/pack.safetensors"));
}

