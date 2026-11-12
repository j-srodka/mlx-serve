//! MiniMax-H3 text-to-audio-video on the media table (`/v1/video/generations`): FL2VA keyframes, REF2VA references,
//! the Turbo LoRA, chained windows and FastH3 Omni's PDD student. The request surface is NARROWER than LTX's: no CFG
//! scale and no pipeline mode, frames on a 17k+5 ladder; anything this backend cannot honor is a named refusal. The
//! result is the response's JSON members (24 fps RGB, 32 kHz stereo PCM16).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ops = @import("../../mlx/ops.zig");
const lora = @import("../../sdk/weights/lora.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const video_out = @import("../../sdk/media/video_out.zig");
const frame_in = @import("../../sdk/media/frame_in.zig");
const wav = @import("../../sdk/media/wav.zig");
const h3 = @import("h3.zig");
const upscaler = @import("upscaler.zig");
const vsa = @import("vsa.zig");
const h3_refmod = @import("../../h3_refmod.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const vae_mod = @import("vae.zig");
const audio_mod = @import("audio.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .video,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .set_loras = &setLoras,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const A = mlx.mlx_array;

pub const Engine = struct {
    model_dir: []u8,
    stream: mlx.mlx_stream,
    owns_stream: bool,
    machine: abi.Machine,
    geometry: h3.Geometry,
    /// The REF2VA partition (the pack's declared tasks; its files are the FL2VA ones).
    supports_refs: bool,
    /// A pack declaring the `fl2va` task (keyframes, pins, `latent_out`). A hybrid pack sets this AND `supports_refs`.
    supports_fl2va: bool,
    /// A pack declaring ref2va without t2va (FastH3 Omni) needs an image or video reference on every request.
    supports_t2va: bool,
    pdd: ?h3.Pdd,
    vsa: ?vsa.Contract,
    /// The text encoder and DiT kept loaded between requests while memory allows.
    resident: h3.Resident,
    /// The request's style LoRAs (`set_loras`), stacked with Turbo inside `generate`.
    lora_paths: [lora.MAX_LORAS][]u8 = undefined,
    lora_scales: [lora.MAX_LORAS]f32 = undefined,
    lora_n: usize = 0,
    result: []u8 = &.{},
    /// `POST /v1/video/refmods` response body (taken by the host after `refmodsExtract`).
    refmods_result: []u8 = &.{},

    fn freeLoras(e: *Engine) void {
        for (e.lora_paths[0..e.lora_n]) |p| gpa.free(p);
        e.lora_n = 0;
    }

    fn path(e: *const Engine, comptime name: []const u8) ![]u8 {
        return std.fmt.allocPrint(gpa, "{s}/" ++ name, .{e.model_dir});
    }

};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    var buf: [1 << 20]u8 = undefined;
    var why: Verdict = .{};
    const cfg = ctx.dir.readFile("config.json", &buf, &why) orelse "{}";
    const fail = struct {
        fn named(o: *Verdict, err: anyerror) ?*anyopaque {
            o.* = Verdict.named(.failed, @errorName(err));
            return null;
        }
    }.named;
    const geometry = config.geometry(gpa, cfg) catch |err| return fail(out, err);
    const pdd = h3.Pdd.fromConfig(gpa, cfg) catch |err| return fail(out, err);
    const contract = vsa.Contract.fromConfig(gpa, cfg) catch |err| return fail(out, err);
    const e = gpa.create(Engine) catch return fail(out, error.OutOfMemory);
    const owns = ctx.stream.ctx == null;
    e.* = .{
        .model_dir = gpa.dupe(u8, ctx.model_dir.slice()) catch {
            gpa.destroy(e);
            return fail(out, error.OutOfMemory);
        },
        .stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx },
        .owns_stream = owns,
        .machine = if (ctx.machine) |m| m.* else .{},
        .geometry = geometry,
        .supports_refs = config.declaresTask(gpa, cfg, "ref2va"),
        .supports_fl2va = config.declaresTask(gpa, cfg, "fl2va"),
        .supports_t2va = config.declaresTask(gpa, cfg, "t2va"),
        .pdd = pdd,
        .vsa = contract,
        .resident = h3.Resident.init(gpa),
    };
    h3.registerResident(&e.resident);
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    h3.unregisterResident(&e.resident);
    _ = e.resident.release();
    e.freeLoras();
    gpa.free(e.result);
    if (e.refmods_result.len != 0) gpa.free(e.refmods_result);
    gpa.free(e.model_dir);
    if (e.owns_stream) _ = mlx.mlx_stream_free(e.stream);
    gpa.destroy(e);
}

/// The request's adapters, proven HERE: the DiT loads per request, so `loadFile`'s own check would be reached minutes
/// into a generation. They stack with Turbo, which `generate` puts first.
fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.freeLoras();
    for (loras.slice()) |l| {
        lora.validatePath(l.path.slice()) catch return Verdict.refuse(.unsupported, "'lora_paths' must be absolute paths to readable .safetensors files");
        e.lora_paths[e.lora_n] = gpa.dupe(u8, l.path.slice()) catch return Verdict.refuse(.busy, "out of memory");
        e.lora_scales[e.lora_n] = l.scale;
        e.lora_n += 1;
    }
    return .ok;
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// `default` when omitted, else held to [lo, hi] before any cast.
fn stepsIn(b: *const job.Body, default: u32, lo: u32, hi: u32) ?u32 {
    const v = b.get("steps") orelse return default;
    const n: i64 = switch (v) {
        .integer => |i| i,
        .null => return default,
        else => return null,
    };
    return if (n < lo or n > hi) null else @intCast(n);
}

fn u32Field(b: *const job.Body, key: []const u8, default: u32) u32 {
    return @intCast(@min(b.uint(key) orelse default, std.math.maxInt(u32)));
}

/// `key` holds a value (absent and null are unset).
fn fieldSet(b: *const job.Body, key: []const u8) bool {
    const v = b.get(key) orelse return false;
    return v != .null;
}

fn pm1Frame(bytes: []const u8, h: u32, w: u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?A {
    return frame_in.pm1(gpa, bytes, h, w, dt, s);
}

const Keyframes = struct {
    buf: [2]h3.Keyframe = undefined,
    n: usize = 0,

    fn deinit(k: *Keyframes) void {
        for (k.buf[0..k.n]) |kf| {
            _ = mlx.mlx_array_free(kf.pixels);
        }
    }
};

/// FL2VA keyframes: first = the geometry anchor (bf16), last = f32, both cover-cropped to the canvas. NOT graceful:
/// the user asked for THIS frame.
fn keyframes(b: *const job.Body, width: u32, height: u32, s: mlx.mlx_stream, out: *Keyframes) ?Verdict {
    inline for (.{ .{ "first_frame_image", h3.KeyframeAnchor.first, mlx.mlx_dtype.bfloat16 }, .{ "last_frame_image", h3.KeyframeAnchor.last, mlx.mlx_dtype.float32 } }) |k| {
        if (b.str(k[0])) |b64| if (b64.len > 0) {
            const bytes = job.imageUpload(gpa, b64) catch return bad("keyframe image is not valid base64");
            defer gpa.free(bytes);
            const undecodable = bad("keyframe image could not be decoded (PNG/JPEG expected)");
            const pixels = pm1Frame(bytes, height, width, k[2], s) orelse return undecodable;
            out.buf[out.n] = .{ .anchor = k[1], .pixels = pixels };
            out.n += 1;
            log.info("[video] minimax-h3 {s} keyframe conditioning engaged ({d}x{d})\n", .{ @tagName(k[1]), width, height });
        };
    }
    return null;
}

/// A reference WAV as the audio VAE reads it: 32 kHz stereo interleaved.
fn wav32kStereo(b64: []const u8) ![]f32 {
    const bytes = try job.base64Decode(gpa, b64);
    defer gpa.free(bytes);
    const dec = try wav.decode(gpa, bytes);
    defer gpa.free(dec.pcm);
    const stereo = try wav.toStereoInterleaved(gpa, dec.pcm, dec.channels);
    defer gpa.free(stereo);
    return wav.resampleLinear(gpa, stereo, 2, dec.sample_rate, @import("audio.zig").SAMPLE_RATE);
}

/// `[2, L]` f32 from interleaved stereo: the channels ride the batch axis.
fn planar(pcm: []const f32) !A {
    const frames = pcm.len / 2;
    const p = try gpa.alloc(f32, pcm.len);
    defer gpa.free(p);
    for (0..frames) |i| {
        p[i] = pcm[i * 2];
        p[frames + i] = pcm[i * 2 + 1];
    }
    return mlx.mlx_array_new_data(p.ptr, &[_]c_int{ 2, @intCast(frames) }, 2, .float32);
}

/// Every `stride`-th frame at `th`x`tw`: `[1,3,T,H,W]` for the VAE (`cat_axis` 2) or `[n,3,H,W]` for the tower (0).
fn videoFrames(frames: []const []u8, th: u32, tw: u32, stride: u32, cat_axis: c_int, s: mlx.mlx_stream) !?A {
    var g = ops.Graph.init(gpa, s);
    defer g.deinit();
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(gpa);
    var i: usize = 0;
    while (i < frames.len) : (i += stride) {
        const f = g.adopt(pm1Frame(frames[i], th, tw, .bfloat16, s) orelse return null) catch return null;
        try parts.append(gpa, if (cat_axis == 2) f else try g.reshape(f, &.{ 1, 3, @intCast(th), @intCast(tw) }));
    }
    if (parts.items.len == 0) return null;
    return g.escape(try g.concat(parts.items, cat_axis));
}

/// The REF2VA references decoded into `out` in `resolveRefs` order; a reference the server cannot use is a named
/// refusal, never dropped (the generation would succeed and ignore what the user asked it to follow).
fn countRefModAudioPacks(items: []const h3.RefMedia) u32 {
    var n: u32 = 0;
    for (items) |m| {
        if (m.latent != null and m.ref.kind == .audio) n += 1;
    }
    return n;
}

fn refModPaths(req: *const abi.Request, b: *const job.Body, s: mlx.mlx_stream, out: *std.ArrayList(h3.RefMedia)) !?Verdict {
    const v = b.get("ref_mod_paths") orelse return null;
    if (v == .null) return null;
    if (v != .array) return bad("'ref_mod_paths' must be a JSON array of absolute .safetensors paths");
    if (v.array.items.len == 0) return null;
    if (v.array.items.len > h3.MAX_REF_TOTAL)
        return req.refuseFmt(.unsupported, "too many 'ref_mod_paths'", "too many 'ref_mod_paths' (max {d})", .{h3.MAX_REF_TOTAL});

    const io = std.Io.Threaded.global_single_threaded.io();
    var n_img: u32 = 0;
    var n_vid: u32 = 0;
    var n_aud: u32 = 0;
    for (v.array.items, 0..) |item, i| {
        if (item != .string) return req.refuseFmt(.unsupported, "a 'ref_mod_paths' entry is not a string", "'ref_mod_paths'[{d}] must be a JSON string", .{i});
        const path = item.string;
        if (!std.mem.endsWith(u8, path, ".safetensors"))
            return req.refuseFmt(.unsupported, "each 'ref_mod_paths' entry must be a .safetensors file", "'ref_mod_paths'[{d}] must end with .safetensors", .{i});
        h3_refmod.validateAbsoluteSafetensorsPath(io, path) catch
            return req.refuseFmt(.unsupported, "'ref_mod_paths' entries must be absolute .safetensors paths", "'ref_mod_paths'[{d}] must be an absolute path to a .safetensors file", .{i});
        var loaded = h3_refmod.load(gpa, io, path, s) catch
            return req.refuseFmt(.unsupported, "could not load a ref mod file", "could not load 'ref_mod_paths'[{d}] (missing latent tensor or refmod_meta)", .{i});
        const ordinal: u32 = switch (loaded.meta.kind) {
            .image => blk: {
                n_img += 1;
                break :blk n_img;
            },
            .video => blk: {
                n_vid += 1;
                break :blk n_vid;
            },
            .audio => blk: {
                n_aud += 1;
                if (n_aud > h3.MAX_REF_AUDIOS)
                    return req.refuseFmt(.unsupported, "too many audio ref mods", "too many audio ref mods (max {d})", .{h3.MAX_REF_AUDIOS});
                break :blk n_aud;
            },
        };
        const r = loaded.meta.toResolvedRef(ordinal, i);
        try out.append(gpa, .{ .ref = r, .latent = loaded.takeLatent() });
        log.info("[minimax-h3] ref mod #{d}: {s} {d}x{d} latent_t={d} path={s}\n", .{
            i + 1,
            @tagName(loaded.meta.kind),
            loaded.meta.canvas_w,
            loaded.meta.canvas_h,
            loaded.meta.latent_t,
            path,
        });
    }
    return null;
}

fn references(
    req: *const abi.Request,
    b: *const job.Body,
    gen_w: u32,
    gen_h: u32,
    gen_frames: u32,
    s: mlx.mlx_stream,
    out: *std.ArrayList(h3.RefMedia),
    limits: h3.RefLimits,
    audio_ordinal_base: u32,
) !?Verdict {
    const mode: h3.RefImageSizing = blk: {
        const raw = b.str("ref_image_size") orelse break :blk .match;
        if (std.mem.eql(u8, raw, "match")) break :blk .match;
        if (std.mem.eql(u8, raw, "max")) break :blk .max;
        return bad("'ref_image_size' must be \"match\" or \"max\"");
    };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var images: std.ArrayList([]u8) = .empty;
    var videos: std.ArrayList([][]u8) = .empty;
    var tracks: std.ArrayList(?[]f32) = .empty;
    var audios: std.ArrayList([]f32) = .empty;
    defer for (tracks.items) |t| if (t) |x| gpa.free(x);
    defer for (audios.items) |x| gpa.free(x);

    if (b.get("ref_images")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
        if (item != .string) return bad("'ref_images' must be an array of base64 PNG/JPEG strings");
        const bytes = job.imageUpload(a, item.string) catch return req.refuseFmt(.unsupported, "a 'ref_images' entry is not valid base64", "'ref_images'[{d}] is not valid base64", .{i});
        try images.append(a, bytes);
    };
    if (b.get("ref_videos")) |v| if (v == .array) for (v.array.items, 0..) |item, vi| {
        if (item != .object) return bad("'ref_videos' must be an array of {\"frames\":[…],\"audio\":\"…\"} objects");
        const fr = item.object.get("frames") orelse return req.refuseFmt(.unsupported, "a 'ref_videos' entry needs a 'frames' array", "'ref_videos'[{d}] needs a 'frames' array of base64 PNG/JPEG strings", .{vi});
        if (fr != .array) return req.refuseFmt(.unsupported, "a 'ref_videos' entry needs a 'frames' array", "'ref_videos'[{d}] needs a 'frames' array of base64 PNG/JPEG strings", .{vi});
        const frames = try a.alloc([]u8, fr.array.items.len);
        for (fr.array.items, frames, 0..) |f, *dst, fi| {
            if (f != .string) return req.refuseFmt(.unsupported, "a 'ref_videos' frames array is malformed", "'ref_videos'[{d}].frames must be an array of base64 PNG/JPEG strings", .{vi});
            dst.* = job.imageUpload(a, f.string) catch return req.refuseFmt(.unsupported, "a 'ref_videos' frame is not valid base64", "'ref_videos'[{d}].frames[{d}] is not valid base64", .{ vi, fi });
        }
        try videos.append(a, frames);
        var track: ?[]f32 = null;
        if (item.object.get("audio")) |au| if (au == .string and au.string.len > 0) {
            track = wav32kStereo(au.string) catch return req.refuseFmt(.unsupported, "a 'ref_videos' soundtrack could not be decoded", "'ref_videos'[{d}].audio must be a PCM16/PCM24/float32 WAV", .{vi});
        };
        try tracks.append(a, track);
    };
    if (b.get("ref_audios")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
        if (item != .string) return bad("'ref_audios' must be an array of base64 WAV strings");
        const pcm = wav32kStereo(item.string) catch return req.refuseFmt(.unsupported, "a 'ref_audios' entry could not be decoded", "'ref_audios'[{d}] must be a PCM16/PCM24/float32 WAV", .{i});
        try audios.append(a, pcm);
    };
    if (images.items.len + videos.items.len + audios.items.len == 0) return null;

    // Source dimensions are all `resolveRefs` sizes from: an unmeasurable reference is refused here, by field.
    const in_i = try a.alloc(h3.RefInput, images.items.len);
    for (images.items, in_i, 0..) |bytes, *r, i| {
        const sz = image_pixels.nativeSize(bytes) orelse
            return req.refuseFmt(.unsupported, "a 'ref_images' entry could not be decoded (PNG/JPEG expected)", "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{i});
        r.* = .{ .kind = .image, .w = sz.w, .h = sz.h };
    }
    const in_v = try a.alloc(h3.RefInput, videos.items.len);
    for (videos.items, tracks.items, in_v, 0..) |fr, track, *r, i| {
        if (fr.len == 0) return req.refuseFmt(.unsupported, "a 'ref_videos' entry has no frames", "'ref_videos'[{d}] has no frames", .{i});
        const sz = image_pixels.nativeSize(fr[0]) orelse return req.refuseFmt(.unsupported, "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)", "could not decode 'ref_videos'[{d}].frames[0] (PNG/JPEG expected)", .{i});
        // A reference longer than the generation is cut before it snaps to the ladder: the output cannot reach it.
        r.* = .{ .kind = .video, .w = sz.w, .h = sz.h, .frames = @min(@as(u32, @intCast(fr.len)), gen_frames), .soundtrack_samples = if (track) |p| @intCast(p.len / 2) else null };
    }
    const in_a = try a.alloc(h3.RefInput, audios.items.len);
    for (audios.items, in_a) |pcm, *r| r.* = .{ .kind = .audio, .audio_samples = @intCast(pcm.len / 2) };

    var res = switch (try h3.resolveRefs(gpa, in_i, in_v, in_a, gen_w, gen_h, mode, limits, audio_ordinal_base)) {
        .ok => |r| r,
        .reject => |why| {
            return switch (why) {
                .too_many_images => req.refuseFmt(.unsupported, "too many reference images", "too many reference images ('ref_images' takes at most {d})", .{limits.max_images}),
                .too_many_videos => req.refuseFmt(.unsupported, "too many reference videos", "too many reference videos ('ref_videos' takes at most {d})", .{limits.max_videos}),
                .too_many_audios => req.refuseFmt(.unsupported, "too many reference audios", "too many reference audios ('ref_audios' takes at most {d})", .{limits.max_audios}),
                .too_many_refs => req.refuseFmt(.unsupported, "too many references", "too many references: at most {d} files across 'ref_images', 'ref_videos' and 'ref_audios' combined", .{limits.max_total}),
                else => Verdict.named(.unsupported, why.message()),
            };
        },
    };
    defer res.deinit();
    // Decoded at the canvases the resolver picked: the VAE canvas for the DiT, the Qwen canvas for the tower.
    for (res.refs) |r| {
        try out.append(gpa, .{ .ref = r });
        const media = &out.items[out.items.len - 1];
        switch (r.kind) {
            .image => {
                const bytes = images.items[r.src_index];
                const vfit = h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const undecodable = req.refuseFmt(.unsupported, "a 'ref_images' entry could not be decoded (PNG/JPEG expected)", "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{r.src_index});
                media.pixels = pm1Frame(bytes, r.canvas.h, r.canvas.w, .bfloat16, s) orelse return undecodable;
                media.vision = pm1Frame(bytes, vfit.h, vfit.w, .bfloat16, s) orelse return undecodable;
            },
            .video => {
                const fr = videos.items[r.src_index];
                const vfit = h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const undecodable = req.refuseFmt(.unsupported, "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)", "could not decode a 'ref_videos'[{d}] frame (PNG/JPEG expected)", .{r.src_index});
                media.pixels = try videoFrames(fr[0..r.frames], r.canvas.h, r.canvas.w, 1, 2, s) orelse return undecodable;
                media.vision = try videoFrames(fr[0..r.frames], vfit.h, vfit.w, h3.qwenFrameStride(), 0, s) orelse return undecodable;
                if (tracks.items[r.src_index]) |pcm| media.waveform = try planar(pcm);
            },
            .audio => media.waveform = try planar(audios.items[r.src_index]),
        }
    }
    return null;
}

/// Live memory: host RAM (what the OS will grant, and the total) from the request's machine, and the GPU's
/// working-set room. Unknown host RAM keeps nothing resident and skips the staged-canvas check.
const Mem = struct { host_avail: u64 = 0, total: u64 = 0, gpu_limit: u64, gpu_active: u64 };

fn memNow(req: *const abi.Request) Mem {
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    const m = req.machine orelse &abi.Machine{};
    return .{ .host_avail = m.ram_available, .total = m.ram_total, .gpu_limit = mlx.maxRecommendedWorkingSet(), .gpu_active = active };
}

/// The resident cache for this request, or null for the staged plan (whatever was held is released first, so the
/// staged peak starts from a clean slate). File sizes are read per request: a file can land while the server is up.
fn residentFor(e: *Engine, dir: *const abi.PackDir, activations: u64, mem: Mem) ?struct { r: *h3.Resident, bytes: u64 } {
    const need = config.residentBytes(dir.fileBytes("text_encoder.safetensors"), config.ditFileBytes(dir, &e.machine), dir.fileBytes("video_vae.safetensors") + dir.fileBytes("audio_vae.safetensors"), dir.fileBytes("turbo_lora.safetensors"), activations);
    const avail = config.availBytes(mem.host_avail, mem.gpu_limit, mem.gpu_active);
    if (!(config.residentEnabled() and !e.machine.ane_video and config.keepResident(avail, e.resident.bytes, need, config.residentMargin(mem.total)))) {
        _ = e.resident.release();
        return null;
    }
    return .{ .r = &e.resident, .bytes = need };
}

const UpscaleSpec = struct { tier: u32 = 1, factor: ?f64 = null };

/// `latent_upscale` `{factor?, tier?}` (the target canvas is the request's width/height).
fn parseUpscale(b: *const job.Body, out: *UpscaleSpec) ?Verdict {
    const uv = b.get("latent_upscale").?;
    if (uv != .object) return bad("'latent_upscale' must be an object with factor or width+height");
    const o = uv.object;
    if (o.get("tier")) |t| {
        const n: i64 = switch (t) {
            .integer => |i| i,
            else => return bad("'latent_upscale.tier' must be an integer 1–6"),
        };
        if (n < 1 or n > 6) return bad("'latent_upscale.tier' must be an integer 1–6");
        out.tier = @intCast(n);
    }
    if (o.get("factor")) |f| {
        out.factor = switch (f) {
            .float => |x| x,
            .integer => |n| @floatFromInt(n),
            else => return bad("'latent_upscale.factor' must be a number in [1, 4]"),
        };
        if (out.factor.? < upscaler.SCALE_MIN or out.factor.? > upscaler.SCALE_MAX)
            return bad("'latent_upscale.factor' must be a number in [1, 4]");
    }
    return null;
}

const UPSCALER_MISSING = "latent_upscale weights not found — pass --h3-upscaler or put a 2D v1 safetensors in the pack's latent_upscaler/";

/// `"refine": false`: the init sidecar's video latent through LatentResizer2D to the target grid, written to
/// `latent_out` with its audio copied through. No DiT and any latent T, so the upscaler's temporal chunks run across a
/// whole run of joined takes. The target is the request's width/height, else the factor times the sidecar's grid.
fn upscaleOnly(e: *Engine, req: *const abi.Request, b: *const job.Body) Verdict {
    const init_raw = b.str("latent_init_path") orelse "";
    const out_raw = b.str("latent_out") orelse "";
    const has_schedule = fieldSet(b, "refine_sigmas") or fieldSet(b, "refine_start_sigma") or fieldSet(b, "refine_audio");
    const has_conditions = for ([_][]const u8{
        "latent_prefix_frames", "latent_suffix_frames", "latent_prefix_path", "latent_suffix_path",
        "ref_images",           "ref_videos",           "ref_audios",         "ref_mod_paths",
        "first_frame_image",    "last_frame_image",     "chain_windows",
    }) |k| {
        if (fieldSet(b, k)) break true;
    } else false;
    if (h3.upscaleOnlyRequestError(init_raw.len > 0, fieldSet(b, "latent_upscale"), out_raw.len > 0, has_schedule, has_conditions)) |msg|
        return Verdict.named(.unsupported, msg);
    var up: UpscaleSpec = .{};
    if (parseUpscale(b, &up)) |v| return v;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (!std.fs.path.isAbsolute(init_raw)) return bad("'latent_init_path' must be an absolute path");
    if (!std.fs.path.isAbsolute(out_raw)) return bad("'latent_out' must be an absolute path");
    const parent = std.fs.path.dirname(out_raw) orelse return bad("'latent_out' must be an absolute path");
    (std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return bad("'latent_out' parent directory does not exist")).close(io);
    const wpath = (upscaler.resolvePath(gpa, e.model_dir) catch return bad(UPSCALER_MISSING)) orelse return bad(UPSCALER_MISSING);
    defer gpa.free(wpath);

    const s = e.stream;
    const init_z = gpa.dupeSentinel(u8, init_raw, 0) catch return job.failure(error.OutOfMemory, "out of memory");
    defer gpa.free(init_z);
    const loaded = h3.loadLatentSidecar(gpa, init_z, s) catch |err| return switch (err) {
        error.BadLatentSidecarPack => bad("'latent_init_path' sidecar must declare pack fl2va"),
        error.MissingVideoLatent => bad("'latent_init_path' sidecar missing video_latent"),
        error.MissingAudioLatent => bad("'latent_init_path' sidecar missing audio_latent"),
        else => bad("'latent_init_path' could not be loaded"),
    };
    defer _ = mlx.mlx_array_free(loaded.video);
    defer _ = mlx.mlx_array_free(loaded.audio);
    const vs = mlx.getShape(loaded.video);
    if (vs.len != 5 or vs[1] != upscaler.IN_CH) return bad("'latent_init_path' video_latent must be [1, 24, T, H, W]");
    const lat_t: u32 = @intCast(vs[2]);
    const src_h: u32 = @intCast(vs[3]);
    const src_w: u32 = @intCast(vs[4]);
    var dst_h: u32 = undefined;
    var dst_w: u32 = undefined;
    if (fieldSet(b, "width") or fieldSet(b, "height")) {
        const width = u32Field(b, "width", 0);
        const height = u32Field(b, "height", 0);
        if (width == 0 or height == 0 or width % 32 != 0 or height % 32 != 0) return bad("width and height must be multiples of 32");
        dst_h = height / h3.VAE_SPATIAL;
        dst_w = width / h3.VAE_SPATIAL;
        if (up.factor) |f| if (@round(@as(f64, @floatFromInt(src_h)) * f) != @as(f64, @floatFromInt(dst_h)) or
            @round(@as(f64, @floatFromInt(src_w)) * f) != @as(f64, @floatFromInt(dst_w)))
            return bad("'latent_upscale.factor' does not match request width/height");
    } else {
        const f = up.factor orelse return bad("refine:false (upscale only) needs width+height or latent_upscale.factor");
        dst_h = @intFromFloat(@round(@as(f64, @floatFromInt(src_h)) * f));
        dst_w = @intFromFloat(@round(@as(f64, @floatFromInt(src_w)) * f));
        if (dst_h % 2 != 0 or dst_w % 2 != 0) return bad("latent_upscale.factor must land on width and height multiples of 32");
    }

    // The upscaler's chunk working set plus the whole run in and out (f32), against live memory.
    const run_bytes = @as(u64, lat_t) * upscaler.IN_CH * 4 * (@as(u64, src_h) * src_w + 2 * @as(u64, dst_h) * dst_w);
    const need = upscaler.workingSetBytes(0, dst_h, dst_w, lat_t) + run_bytes;
    const mem = memNow(req);
    const avail = config.availBytes(mem.host_avail, mem.gpu_limit, mem.gpu_active);
    if (avail > 0 and need > avail and !abi.skipMemPreflight(req.machine orelse &abi.Machine{})) {
        const gb = 1024.0 * 1024.0 * 1024.0;
        return req.refuseFmt(.unsupported, "the latent upscale does not fit in memory: upscale fewer frames", "latent upscale of {d} latent frames to {d}x{d} needs ~{d:.1} GB but {d:.1} GB is available: upscale fewer frames", .{
            lat_t, dst_w * h3.VAE_SPATIAL, dst_h * h3.VAE_SPATIAL, @as(f64, @floatFromInt(need)) / gb, @as(f64, @floatFromInt(avail)) / gb,
        });
    }

    var um = upscaler.Model.load(gpa, wpath) catch return bad("latent_upscale weights could not be loaded");
    defer um.deinit();
    const scaled = um.forward(gpa, loaded.video, dst_h, dst_w, up.tier, s) catch |err| return job.failure(err, "MiniMax-H3 latent upscale failed");
    defer _ = mlx.mlx_array_free(scaled);
    const out_z = gpa.dupeSentinel(u8, out_raw, 0) catch return job.failure(error.OutOfMemory, "out of memory");
    defer gpa.free(out_z);
    h3.saveLatentSidecar(out_z, scaled, loaded.audio, s) catch return Verdict.refuse(.failed, "MiniMax-H3 failed to write latent_out");
    log.info("[minimax-h3] latent upscale only: {d} latent frames {d}x{d} -> {d}x{d} (tier {d}), audio carried\n", .{ lat_t, src_w, src_h, dst_w, dst_h, up.tier });

    var esc: std.ArrayList(u8) = .empty;
    defer esc.deinit(gpa);
    appendJsonEscaped(gpa, &esc, out_raw) catch return job.failure(error.OutOfMemory, "out of memory");
    const members = std.fmt.allocPrint(gpa, "\"latent_out\":\"{s}\",\"latent_t\":{d},\"width\":{d},\"height\":{d}", .{ esc.items, lat_t, dst_w * h3.VAE_SPATIAL, dst_h * h3.VAE_SPATIAL }) catch
        return job.failure(error.OutOfMemory, "out of memory");
    gpa.free(e.result);
    e.result = members;
    return .ok;
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const dir = req.dir orelse return Verdict.refuse(.failed, "minimax_h3: the host gave no pack dir");
    var b = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer b.deinit();
    if (b.get("refine")) |v| {
        if (v != .bool) return bad("'refine' must be true or false");
        if (!v.bool) return upscaleOnly(e, req, &b);
    }
    const prompt = b.str("prompt") orelse return bad("missing 'prompt'");
    if (prompt.len == 0) return bad("empty 'prompt'");
    // `cfg_scale` / `stg_scale` / `pipeline` are ignored, not refused: the app sends them for every video backend and
    // H3 is CFG-distilled and single-pipeline, so there is nothing they could have selected.

    const turbo = b.flag("turbo");
    const pdd_steps: ?u32 = if (e.pdd) |p| p.steps() else null;
    if (turbo and pdd_steps != null) return bad("'turbo' is the base model's few-step adapter — this pack is already a distilled student and runs its own trained steps");
    if (turbo and dir.fileBytes("turbo_lora.safetensors") == 0) return bad("this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors");

    const width = u32Field(&b, "width", 256);
    const height = u32Field(&b, "height", 256);
    const steps = if (pdd_steps) |n|
        stepsIn(&b, n, n, n) orelse return req.refuseFmt(.unsupported, "this pack runs a fixed step count — omit 'steps'", "this pack is a distilled student that runs exactly {d} steps — omit 'steps' or send {d}", .{ n, n })
    else
        stepsIn(&b, if (turbo) 4 else 30, 1, 100) orelse return bad("'steps' must be in [1,100]");
    const seed = b.uint("seed") orelse 0;
    const requested_frames = u32Field(&b, "num_frames", 56);
    if (width % 32 != 0 or height % 32 != 0) return bad("width and height must be multiples of 32");
    if (turbo and steps < 4) return bad("turbo needs at least 4 steps (the distillation's own floor, and its default)");

    // Chained windows ride FL2VA keyframe conditioning: `num_frames` is per window.
    const chain_windows = u32Field(&b, "chain_windows", 1);
    if (chain_windows < 1 or chain_windows > 6) return bad("chain_windows must be 1-6");
    if (chain_windows > 1 and e.supports_refs and !e.supports_fl2va) return bad("chained windows ride FL2VA keyframe conditioning — the REF2VA pack cannot serve them; load an FL2VA checkpoint");

    const latent_prefix_frames = u32Field(&b, "latent_prefix_frames", 0);
    const latent_suffix_frames = u32Field(&b, "latent_suffix_frames", 0);
    const latent_out_raw = b.str("latent_out");
    const latent_prefix_path_raw = b.str("latent_prefix_path");
    const latent_suffix_path_raw = b.str("latent_suffix_path");
    const latent_init_raw = b.str("latent_init_path");
    const has_latent_init = latent_init_raw != null and latent_init_raw.?.len > 0;
    const has_latent_upscale = fieldSet(&b, "latent_upscale");
    const refine_audio = b.flag("refine_audio");
    var refine_sigmas_buf: [16]f64 = undefined;
    var refine_sigmas: ?[]const f64 = null;
    if (b.get("refine_sigmas")) |v| {
        if (v != .array) return bad("'refine_sigmas' must be an array of descending sigmas ending at 0");
        if (v.array.items.len < 2 or v.array.items.len > refine_sigmas_buf.len)
            return bad("'refine_sigmas' must have 2–16 values ending at 0");
        for (v.array.items, 0..) |item, i| {
            refine_sigmas_buf[i] = switch (item) {
                .float => |f| f,
                .integer => |n| @floatFromInt(n),
                else => return bad("'refine_sigmas' must be an array of descending sigmas ending at 0"),
            };
        }
        refine_sigmas = refine_sigmas_buf[0..v.array.items.len];
        if (h3.refineSigmasError(refine_sigmas.?)) |msg| return Verdict.named(.unsupported, msg);
    } else if (b.float("refine_start_sigma")) |start| {
        const mapped = h3.refineScheduleFromStart(gpa, start, steps, h3.SIGMA_SHIFT_VIDEO) catch return bad("'refine_start_sigma' must be in (0, 1]");
        defer gpa.free(mapped);
        if (mapped.len > refine_sigmas_buf.len) return bad("'refine_sigmas' must have 2–16 values ending at 0");
        @memcpy(refine_sigmas_buf[0..mapped.len], mapped);
        refine_sigmas = refine_sigmas_buf[0..mapped.len];
        if (h3.refineSigmasError(refine_sigmas.?)) |msg| return Verdict.named(.unsupported, msg);
    }
    if ((refine_sigmas != null or refine_audio or has_latent_upscale) and !has_latent_init)
        return bad("latent_init_path is required for refine / latent_upscale");
    var up: UpscaleSpec = .{};
    if (has_latent_upscale) if (parseUpscale(&b, &up)) |v| return v;
    const up_tier = up.tier;
    const up_factor = up.factor;
    const has_latent_out = latent_out_raw != null and latent_out_raw.?.len > 0;
    const has_latent_prefix_path = latent_prefix_path_raw != null and latent_prefix_path_raw.?.len > 0;
    const has_latent_suffix_path = latent_suffix_path_raw != null and latent_suffix_path_raw.?.len > 0;

    // Snapped to the model's own ladder, and said so: a silently different length drifts the client's audio mux.
    const shape = h3.temporalShape(requested_frames);
    if (h3.latentPrefixRequestError(latent_prefix_frames, chain_windows, shape.frame_count, e.supports_fl2va, has_latent_prefix_path)) |msg| return Verdict.named(.unsupported, msg);
    if (h3.latentSuffixRequestError(latent_suffix_frames, chain_windows, shape.frame_count, e.supports_fl2va, has_latent_suffix_path)) |msg| return Verdict.named(.unsupported, msg);
    if (h3.latentPinWindowError(latent_prefix_frames, latent_suffix_frames, shape.frame_count)) |msg| return Verdict.named(.unsupported, msg);
    if (has_latent_prefix_path and latent_prefix_frames == 0) return bad("latent_prefix_path requires latent_prefix_frames");
    if (has_latent_suffix_path and latent_suffix_frames == 0) return bad("latent_suffix_path requires latent_suffix_frames");
    if (has_latent_out and !e.supports_fl2va) return bad("latent_out is FL2VA only — load an FL2VA checkpoint");
    if (has_latent_out and shape.frame_count == 1) return bad("num_frames 1 is a still and cannot use latent_out");
    if (has_latent_prefix_path and !e.supports_fl2va) return bad("latent_prefix_path is FL2VA only — load an FL2VA checkpoint");
    if (has_latent_prefix_path and shape.frame_count == 1) return bad("num_frames 1 is a still and cannot use latent_prefix_path");
    if (has_latent_suffix_path and !e.supports_fl2va) return bad("latent_suffix_path is FL2VA only — load an FL2VA checkpoint");
    if (has_latent_suffix_path and shape.frame_count == 1) return bad("num_frames 1 is a still and cannot use latent_suffix_path");

    const overlap: u32 = if (latent_prefix_frames > 0) latent_prefix_frames else 0;
    const delivered = h3.chainDeliveredFrames(chain_windows, shape.frame_count, overlap);
    const rgb = video_out.rgbBytes(delivered, width, height);
    if (rgb > video_out.MAX_RGB_BYTES) return req.refuseFmt(.unsupported, video_out.TOO_LARGE, video_out.TOO_LARGE_FMT, .{ delivered, width, height, rgb / (1024 * 1024), video_out.MAX_RGB_BYTES / (1024 * 1024) });

    const s = e.stream;
    var kfs: Keyframes = .{};
    defer kfs.deinit();
    if (keyframes(&b, width, height, s, &kfs)) |v| return v;

    var refs: std.ArrayList(h3.RefMedia) = .empty;
    defer {
        for (refs.items) |*m| m.deinit();
        refs.deinit(gpa);
    }
    const has_live_refs = fieldSet(&b, "ref_images") or fieldSet(&b, "ref_videos") or fieldSet(&b, "ref_audios");
    const has_ref_mod_paths = fieldSet(&b, "ref_mod_paths");
    if (has_ref_mod_paths and fieldSet(&b, "ref_images"))
        return Verdict.named(.unsupported, "do not mix 'ref_mod_paths' with ref_images");
    if ((has_live_refs or has_ref_mod_paths) and !e.supports_refs)
        return bad("this MiniMax-H3 pack does not support references (it declares no 'ref2va' task) — load a REF2VA checkpoint");
    const has_refs_intent = has_live_refs or has_ref_mod_paths;
    const ref_pins_ok = e.supports_fl2va and e.supports_refs;
    if (h3.refCompatError(has_refs_intent, chain_windows, latent_prefix_frames, latent_suffix_frames, ref_pins_ok)) |msg|
        return Verdict.named(.unsupported, msg);
    const io = std.Io.Threaded.global_single_threaded.io();
    if (has_latent_out) {
        const raw = latent_out_raw.?;
        if (!std.fs.path.isAbsolute(raw)) return bad("'latent_out' must be an absolute path");
        const parent = std.fs.path.dirname(raw) orelse return bad("'latent_out' must be an absolute path");
        const d = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return bad("'latent_out' parent directory does not exist");
        d.close(io);
    }
    if (has_ref_mod_paths) if (refModPaths(req, &b, s, &refs) catch |err| return job.failure(err, "MiniMax-H3 generation failed")) |v| return v;
    const audio_ref_mods = countRefModAudioPacks(refs.items);
    if (has_live_refs) if (references(req, &b, width, height, shape.frame_count, s, &refs, h3.RefLimits.generation, audio_ref_mods) catch |err| return job.failure(err, "MiniMax-H3 generation failed")) |v| return v;
    if (refs.items.len > h3.MAX_REF_TOTAL)
        return req.refuseFmt(.unsupported, "too many references", "too many references: at most {d} files across live refs and ref_mod_paths combined", .{h3.MAX_REF_TOTAL});
    const has_refs = has_refs_intent or refs.items.len > 0;
    const has_pins = latent_prefix_frames > 0 or latent_suffix_frames > 0;
    if (h3.refineRequestError(has_latent_init, has_latent_upscale, e.supports_fl2va, chain_windows, has_refs, has_pins, e.pdd != null)) |msg|
        return Verdict.named(.unsupported, msg);
    var kf_has_first = false;
    var kf_has_last = false;
    for (kfs.buf[0..kfs.n]) |kf| switch (kf.anchor) {
        .first => kf_has_first = true,
        .last => kf_has_last = true,
    };
    if (h3.keyframeSidecarBindingError(kf_has_first, kf_has_last, has_latent_prefix_path, has_latent_suffix_path)) |msg| return Verdict.named(.unsupported, msg);
    if (e.supports_refs and !e.supports_t2va) {
        const visual = for (refs.items) |m| {
            if (m.ref.kind != .audio) break true;
        } else false;
        if (!visual) return bad("this pack generates from references — attach at least one image or video ('ref_images' / 'ref_videos')");
    }

    // The request's own activations price the resident plan; when it does not fit, the staged DiT stage is checked
    // here, before any stage loads, since a Metal OOM mid-step is uncatchable.
    const distilled = turbo or pdd_steps != null;
    const fast: ?bool = if (b.get("fast")) |v| (if (v == .bool) v.bool else null) else null;
    const speed = h3.resolveSpeed(fast, distilled);
    const bcast_row: u64 = if (speed.bcast_k > 0) @as(u64, e.geometry.dit.num_layers) * e.geometry.dit.hidden_size * 2 else 0;
    var ref_blocks_buf: [h3.MAX_REF_TOTAL]h3.RefBlock = undefined;
    const ref_blocks = blk: {
        if (refs.items.len == 0) break :blk @as([]const h3.RefBlock, &.{});
        for (refs.items, 0..) |m, i| ref_blocks_buf[i] = h3.refBlockFor(m.ref);
        break :blk ref_blocks_buf[0..refs.items.len];
    };
    const rows = config.requestRows(width, height, shape.frame_count, @intCast(kfs.n)) + h3.refCondRows(ref_blocks);
    const activations = config.activationBytes(rows, bcast_row);
    const mem = memNow(req);
    const plan = residentFor(e, dir, activations, mem);
    if (plan == null) {
        const dit = config.ditResidentBytes(config.ditFileBytes(dir, &e.machine), true) + (if (turbo) dir.fileBytes("turbo_lora.safetensors") else 0);
        const bill = config.canvasBill(dit, activations, mem.host_avail, mem.gpu_limit, mem.gpu_active, mlx.cudaAvailable());
        const up_ws = if (has_latent_upscale) upscaler.workingSetBytes(0, height / h3.VAE_SPATIAL, width / h3.VAE_SPATIAL, shape.latent_t) else 0;
        const need = bill.need + up_ws;
        const skip_bill = abi.skipMemPreflight(req.machine orelse &abi.Machine{});
        if (config.refuseStagedCanvasBill(dit, need, bill.avail, skip_bill)) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            const cache = bcast_row > 0;
            return req.refuseFmt(.unsupported, "the canvas does not fit in memory: use a smaller canvas or fewer frames", "{d}x{d} x {d} frames needs ~{d:.1} GB ({d:.1} GB DiT{s} + {d:.1} GB for {d} sequence rows{s}) but {d:.1} GB is available: use a smaller canvas or fewer frames{s}", .{
                width,                                              height,                                    shape.frame_count,
                @as(f64, @floatFromInt(need)) / gb,                 @as(f64, @floatFromInt(dit)) / gb,         if (mlx.cudaAvailable()) " paged to RAM" else "",
                @as(f64, @floatFromInt(activations)) / gb,          rows,                                      if (cache) " with the fast recipe's attention cache" else "",
                @as(f64, @floatFromInt(bill.avail)) / gb,           if (cache) ", or \"turbo\", which keeps no attention cache" else "",
            });
        }
        if (skip_bill and dit > 0 and bill.avail > 0 and need > bill.avail) {
            const gb: f64 = 1024.0 * 1024.0 * 1024.0;
            log.warn("[minimax-h3] canvas bill skipped ({d}x{d} x {d}f billed ~{d:.1} GB, {d:.1} GB available): --skip-mem-preflight — uncatchable OOM mid-step is possible\n", .{
                width, height, shape.frame_count,
                @as(f64, @floatFromInt(need)) / gb, @as(f64, @floatFromInt(bill.avail)) / gb,
            });
        }
    }

    const paths: h3.GenPaths = .{
        .tokenizer_dir = e.model_dir,
        .text_encoder = e.path("text_encoder.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .dit = e.path("transformer.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .vae = e.path("video_vae.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .audio_vae = if (dir.fileBytes("audio_vae.safetensors") > 0) e.path("audio_vae.safetensors") catch null else null,
        .turbo_lora = e.path("turbo_lora.safetensors") catch null,
    };
    defer for ([_]?[]const u8{ paths.text_encoder, paths.dit, paths.vae, paths.audio_vae, paths.turbo_lora }) |p| if (p) |x| gpa.free(x);

    var paths_c: [lora.MAX_LORAS][]const u8 = undefined;
    for (e.lora_paths[0..e.lora_n], 0..) |p, i| paths_c[i] = p;
    if (has_latent_prefix_path) {
        const raw = latent_prefix_path_raw.?;
        if (!std.fs.path.isAbsolute(raw)) return bad("'latent_prefix_path' must be an absolute path");
        const f = std.Io.Dir.openFileAbsolute(io, raw, .{}) catch return bad("'latent_prefix_path' file not found or not readable");
        f.close(io);
    }
    if (has_latent_suffix_path) {
        const raw = latent_suffix_path_raw.?;
        if (!std.fs.path.isAbsolute(raw)) return bad("'latent_suffix_path' must be an absolute path");
        const f = std.Io.Dir.openFileAbsolute(io, raw, .{}) catch return bad("'latent_suffix_path' file not found or not readable");
        f.close(io);
    }
    if (has_latent_init) {
        const raw = latent_init_raw.?;
        if (!std.fs.path.isAbsolute(raw)) return bad("'latent_init_path' must be an absolute path");
        const f = std.Io.Dir.openFileAbsolute(io, raw, .{}) catch return bad("'latent_init_path' file not found or not readable");
        f.close(io);
    }

    var prefix_v: ?mlx.mlx_array = null;
    defer if (prefix_v) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var prefix_a: ?mlx.mlx_array = null;
    defer if (prefix_a) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var suffix_v: ?mlx.mlx_array = null;
    defer if (suffix_v) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var suffix_a: ?mlx.mlx_array = null;
    defer if (suffix_a) |x| {
        _ = mlx.mlx_array_free(x);
    };
    if (has_latent_prefix_path) {
        const pathz = gpa.dupeSentinel(u8, latent_prefix_path_raw.?, 0) catch return job.failure(error.OutOfMemory, "out of memory");
        defer gpa.free(pathz);
        const loaded = h3.loadLatentSidecar(gpa, pathz, s) catch |err| return switch (err) {
            error.BadLatentSidecarPack => bad("'latent_prefix_path' sidecar must declare pack fl2va"),
            error.MissingVideoLatent => bad("'latent_prefix_path' sidecar missing video_latent"),
            error.MissingAudioLatent => bad("'latent_prefix_path' sidecar missing audio_latent"),
            else => bad("'latent_prefix_path' could not be loaded"),
        };
        prefix_v = loaded.video;
        prefix_a = loaded.audio;
    }
    if (has_latent_suffix_path) {
        const pathz = gpa.dupeSentinel(u8, latent_suffix_path_raw.?, 0) catch return job.failure(error.OutOfMemory, "out of memory");
        defer gpa.free(pathz);
        const loaded = h3.loadLatentSidecar(gpa, pathz, s) catch |err| return switch (err) {
            error.BadLatentSidecarPack => bad("'latent_suffix_path' sidecar must declare pack fl2va"),
            error.MissingVideoLatent => bad("'latent_suffix_path' sidecar missing video_latent"),
            error.MissingAudioLatent => bad("'latent_suffix_path' sidecar missing audio_latent"),
            else => bad("'latent_suffix_path' could not be loaded"),
        };
        suffix_v = loaded.video;
        suffix_a = loaded.audio;
    }

    var init_v: ?mlx.mlx_array = null;
    defer if (init_v) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var init_a: ?mlx.mlx_array = null;
    defer if (init_a) |x| {
        _ = mlx.mlx_array_free(x);
    };
    if (has_latent_init) {
        const pathz = gpa.dupeSentinel(u8, latent_init_raw.?, 0) catch return job.failure(error.OutOfMemory, "out of memory");
        defer gpa.free(pathz);
        const loaded = h3.loadLatentSidecar(gpa, pathz, s) catch |err| return switch (err) {
            error.BadLatentSidecarPack => bad("'latent_init_path' sidecar must declare pack fl2va"),
            error.MissingVideoLatent => bad("'latent_init_path' sidecar missing video_latent"),
            error.MissingAudioLatent => bad("'latent_init_path' sidecar missing audio_latent"),
            else => bad("'latent_init_path' could not be loaded"),
        };
        init_v = loaded.video;
        init_a = loaded.audio;
        if (has_latent_upscale) {
            const wpath = (upscaler.resolvePath(gpa, e.model_dir) catch return bad(UPSCALER_MISSING)) orelse return bad(UPSCALER_MISSING);
            defer gpa.free(wpath);
            var um = upscaler.Model.load(gpa, wpath) catch return bad("latent_upscale weights could not be loaded");
            defer um.deinit();
            const iv = init_v.?;
            const vsh = mlx.getShape(iv);
            const src_h: u32 = @intCast(vsh[3]);
            const src_w: u32 = @intCast(vsh[4]);
            const dst_h = height / h3.VAE_SPATIAL;
            const dst_w = width / h3.VAE_SPATIAL;
            if (up_factor) |f| {
                const want_h = @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src_h)) * f)));
                const want_w = @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(src_w)) * f)));
                if (want_h != dst_h or want_w != dst_w)
                    return bad("'latent_upscale.factor' does not match request width/height");
            }
            const scaled = um.forward(gpa, iv, dst_h, dst_w, up_tier, s) catch |err| return job.failure(err, "MiniMax-H3 latent upscale failed");
            _ = mlx.mlx_array_free(iv);
            init_v = scaled;
        }
        const vs = mlx.getShape(init_v.?);
        const as = mlx.getShape(init_a.?);
        if (h3.refineInitGeometryError(vs.len, @intCast(vs[1]), @intCast(vs[2]), @intCast(vs[3]), @intCast(vs[4]), @intCast(as[3]), width, height, shape.frame_count)) |msg|
            return Verdict.named(.unsupported, msg);
    }

    var res = h3.generate(gpa, io, paths, .{
        .geometry = e.geometry,
        .resident = if (plan) |p| p.r else null,
        .resident_bytes = if (plan) |p| p.bytes else 0,
        .prompt = prompt,
        .width = width,
        .height = height,
        .frames = requested_frames,
        .steps = steps,
        .seed = seed,
        .speed = speed,
        .turbo = turbo,
        .pdd = e.pdd,
        .vsa = e.vsa,
        .lora_paths = paths_c[0..e.lora_n],
        .lora_scales = e.lora_scales[0..e.lora_n],
        .chain_windows = chain_windows,
        .latent_prefix_frames = latent_prefix_frames,
        .chain_prefix_video_latent = prefix_v,
        .chain_prefix_audio_latent = prefix_a,
        .latent_suffix_frames = latent_suffix_frames,
        .chain_suffix_video_latent = suffix_v,
        .chain_suffix_audio_latent = suffix_a,
        .init_video_latent = init_v,
        .init_audio_latent = init_a,
        .refine_sigmas = refine_sigmas,
        .refine_audio = refine_audio,
        .keyframes = kfs.buf[0..kfs.n],
        .refs = refs.items,
    }, job.progressOf(progress), s) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        // The LoRA failures are the user's to fix, each with its own remedy.
        error.LoraNoMatch => bad("a LoRA has no modules matching MiniMax-H3's DiT — wrong architecture for this adapter?"),
        error.BadLoraPath => bad("'lora_paths' must be absolute paths to .safetensors files"),
        error.TooManyLoras => bad("too many LoRA adapters (max 8, and turbo takes one of the slots)"),
        error.TurboLoraIncomplete => bad("turbo_lora.safetensors ships modules MiniMax-H3's DiT does not have — replace it with minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora"),
        else => job.failure(err, "MiniMax-H3 generation failed"),
    };
    defer res.deinit();
    if (has_latent_out) {
        const vl = res.video_latent orelse return job.failure(error.MissingVideoLatent, "MiniMax-H3 failed to write latent_out");
        const pathz = gpa.dupeSentinel(u8, latent_out_raw.?, 0) catch return job.failure(error.OutOfMemory, "out of memory");
        defer gpa.free(pathz);
        h3.saveLatentSidecar(pathz, vl, res.audio_latent, s) catch return Verdict.refuse(.failed, "MiniMax-H3 failed to write latent_out");
    }
    const members = encodeResult(&res, s) catch |err| return job.failure(err, "MiniMax-H3 generation failed");
    gpa.free(e.result);
    e.result = members;
    return .ok;
}

/// [1,3,F,H,W] in [-1,1] as RGB8 frames, and the soundtrack as PCM16 when the pack has an audio VAE.
fn encodeResult(res: *const h3.GenResult, s: mlx.mlx_stream) ![]u8 {
    const rgb = try h3.pixelsToRgb8(gpa, res, s);
    defer gpa.free(rgb);
    var pcm: ?[]u8 = null;
    defer if (pcm) |p| gpa.free(p);
    if (res.audio) |wave| pcm = try h3.audioToPcm16(gpa, wave, s);
    const audio: ?video_out.Audio = if (pcm) |p| .{ .sample_rate = @import("audio.zig").SAMPLE_RATE, .channels = 2, .pcm16 = p } else null;
    return video_out.members(gpa, res.frame_count, res.height, res.width, h3.FPS, rgb, audio);
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.result.len == 0) return Verdict.refuse(.failed, "minimax_h3: no finished job to encode");
    out.* = .of(e.result);
    return .ok;
}

fn voiceMaxStereoSamples(seconds: f64) u32 {
    const s = if (seconds <= 0.0) 0.0 else seconds;
    const capped = @min(s, 3600.0);
    return @intFromFloat(@ceil(capped * 32000.0));
}

fn mlxSliceAudioLatentT(z: mlx.mlx_array, t_keep: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    const st = [_]c_int{ 0, 0, 0, 0 };
    const en = [_]c_int{ 1, @intCast(h3_refmod.AUDIO_LATENT_C), @intCast(h3_refmod.AUDIO_STEREO), @intCast(t_keep) };
    const one = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&out, z, &st, 4, &en, 4, &one, 4, s));
    return out;
}

fn appendJsonEscaped(allocator: std.mem.Allocator, w: *std.ArrayList(u8), raw: []const u8) !void {
    for (raw) |c| switch (c) {
        '\\' => try w.appendSlice(allocator, "\\\\"),
        '"' => try w.appendSlice(allocator, "\\\""),
        '\n' => try w.appendSlice(allocator, "\\n"),
        '\r' => try w.appendSlice(allocator, "\\r"),
        '\t' => try w.appendSlice(allocator, "\\t"),
        0...8, 11, 12, 14...31 => {
            var buf: [6]u8 = undefined;
            const s = try std.fmt.bufPrint(&buf, "\\u{d:0>4}", .{c});
            try w.appendSlice(allocator, s);
        },
        else => try w.append(allocator, c),
    };
}

fn refmodsOkJson(allocator: std.mem.Allocator, path: []const u8, kind: []const u8, latent_t: u32) ![]u8 {
    var esc: std.ArrayList(u8) = .empty;
    defer esc.deinit(allocator);
    try appendJsonEscaped(allocator, &esc, path);
    return std.fmt.allocPrint(allocator, "{{\"path\":\"{s}\",\"kind\":\"{s}\",\"latent_t\":{d}}}", .{ esc.items, kind, latent_t });
}

/// Named refusal for visual refmod extract canvas (`null` = ok). Voice extract ignores width/height.
fn refmodsCanvasError(has_voice: bool, width: u32, height: u32) ?[]const u8 {
    if (has_voice) return null;
    if (width == 0 or height == 0) return "missing 'width' or 'height'";
    if (width % 32 != 0 or height % 32 != 0) return "width and height must be multiples of 32";
    return null;
}

fn runRefmodsExtract(e: *Engine, req: *const abi.Request) Verdict {
    if (e.refmods_result.len != 0) {
        gpa.free(e.refmods_result);
        e.refmods_result = &.{};
    }

    var b = job.Body.parse(gpa, req.body.slice()) orelse
        return req.refuseFmt(.unsupported, "invalid JSON body", "invalid JSON body", .{});
    defer b.deinit();

    const out_path = b.str("out_path") orelse
        return req.refuseFmt(.unsupported, "missing 'out_path'", "missing 'out_path'", .{});

    const io = std.Io.Threaded.global_single_threaded.io();
    h3_refmod.validateSafetensorsOutPath(io, out_path) catch
        return req.refuseFmt(.unsupported, "'out_path' must be an absolute path to a new or writable .safetensors file in an existing directory", "'out_path' must be an absolute path to a new or writable .safetensors file in an existing directory", .{});

    const width = u32Field(&b, "width", 0);
    const height = u32Field(&b, "height", 0);
    const has_voice = fieldSet(&b, "voice_max_seconds");
    if (refmodsCanvasError(has_voice, width, height)) |msg| return Verdict.named(.unsupported, msg);

    const s = e.stream;
    const stills = fieldSet(&b, "ref_images");
    const one_video = fieldSet(&b, "ref_videos");
    const audio_list = fieldSet(&b, "ref_audios");

    if (has_voice) {
        if (stills or one_video) return req.refuseFmt(.unsupported, "'voice_max_seconds' uses standalone voice: send 'ref_audios' only, not stills or video", "'voice_max_seconds' uses standalone voice: send 'ref_audios' only, not stills or video", .{});
        if (!audio_list) return req.refuseFmt(.unsupported, "missing 'ref_audios' for voice extract", "missing 'ref_audios' for voice extract", .{});
        const vm = b.float("voice_max_seconds") orelse
            return req.refuseFmt(.unsupported, "'voice_max_seconds' must be a positive number", "'voice_max_seconds' must be a positive number", .{});
        if (vm <= 0.0)
            return req.refuseFmt(.unsupported, "'voice_max_seconds' must be positive", "'voice_max_seconds' must be positive", .{});

        var audios: std.ArrayList([]f32) = .empty;
        defer {
            for (audios.items) |x| gpa.free(x);
            audios.deinit(gpa);
        }
        const rau = b.get("ref_audios") orelse return req.refuseFmt(.unsupported, "missing 'ref_audios' for voice extract", "missing 'ref_audios' for voice extract", .{});
        if (rau != .array) return req.refuseFmt(.unsupported, "'ref_audios' must be an array of base64 WAV strings", "'ref_audios' must be an array of base64 WAV strings", .{});
        for (rau.array.items, 0..) |item, i| {
            if (item != .string) return req.refuseFmt(.unsupported, "'ref_audios' must be an array of base64 WAV strings", "'ref_audios'[{d}] must be a base64 WAV string", .{i});
            const pcm = wav32kStereo(item.string) catch return req.refuseFmt(.unsupported, "a 'ref_audios' entry could not be decoded", "'ref_audios'[{d}] must be a PCM16/PCM24/float32 WAV", .{i});
            audios.append(gpa, pcm) catch return Verdict.refuse(.busy, "out of memory");
        }
        if (audios.items.len != 1)
            return req.refuseFmt(.unsupported, "voice extract expects exactly one 'ref_audios' entry", "voice extract expects exactly one 'ref_audios' entry", .{});

        const apack = e.path("audio_vae.safetensors") catch return Verdict.refuse(.busy, "out of memory");
        defer gpa.free(apack);
        var aw = model_mod.loadWeightsSingleFile(gpa, apack) catch return req.refuseFmt(.unsupported, "this pack has no audio_vae.safetensors", "this pack has no audio_vae.safetensors", .{});
        defer aw.deinit();

        const wf = planar(audios.items[0]) catch return Verdict.refuse(.busy, "out of memory");
        defer _ = mlx.mlx_array_free(wf);

        var z = audio_mod.encode(gpa, &aw, wf, s) catch return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: audio encode failed");
        const max_samples = voiceMaxStereoSamples(vm);
        const wf_shp = mlx.getShape(wf);
        const cur_len: u32 = @intCast(wf_shp[1]);
        const want_frames = audio_mod.latentFrames(@min(max_samples, cur_len));
        const z_shp = mlx.getShape(z);
        const cur_t: u32 = @intCast(z_shp[3]);
        if (want_frames < cur_t) {
            const z2 = mlxSliceAudioLatentT(z, want_frames, s) catch {
                _ = mlx.mlx_array_free(z);
                return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: audio latent trim failed");
            };
            _ = mlx.mlx_array_free(z);
            z = z2;
        }

        const meta = h3_refmod.metaFromAudioLatent(z) catch {
            _ = mlx.mlx_array_free(z);
            return Verdict.refuse(.failed, "bad audio latent shape");
        };
        h3_refmod.save(gpa, out_path, meta, z) catch |err| {
            _ = mlx.mlx_array_free(z);
            return switch (err) {
                error.OutOfMemory => Verdict.refuse(.busy, "out of memory"),
                else => Verdict.named(.failed, @errorName(err)),
            };
        };
        _ = mlx.mlx_array_free(z);

        e.refmods_result = refmodsOkJson(gpa, out_path, "audio", meta.latent_t) catch return Verdict.refuse(.busy, "out of memory");
        return .ok;
    }

    if (stills == one_video)
        return req.refuseFmt(.unsupported, "send either 'ref_images' OR 'ref_videos' (stills XOR one video reference)", "send either 'ref_images' OR 'ref_videos' (stills XOR one video reference), not both nor neither", .{});
    if (audio_list)
        return req.refuseFmt(.unsupported, "standalone 'ref_audios' requires 'voice_max_seconds' (voice extract)", "standalone 'ref_audios' requires 'voice_max_seconds' (voice extract)", .{});

    const gen_frames = h3.temporalShape(124).frame_count;
    var refs: std.ArrayList(h3.RefMedia) = .empty;
    defer {
        for (refs.items) |*m| m.deinit();
        refs.deinit(gpa);
    }
    if (references(req, &b, width, height, gen_frames, s, &refs, h3.RefLimits.refmods_visual, 0) catch |err| return job.failure(err, "MiniMax-H3 refmods: reference decode failed")) |v| return v;
    if (refs.items.len == 0)
        return req.refuseFmt(.unsupported, "attach reference images or one reference video", "attach 'ref_images' or one 'ref_videos' entry", .{});

    const vae_path = e.path("video_vae.safetensors") catch return Verdict.refuse(.busy, "out of memory");
    defer gpa.free(vae_path);
    var vw = model_mod.loadWeightsSingleFile(gpa, vae_path) catch return job.failure(error.FileNotFound, "MiniMax-H3 refmods: video VAE missing");
    defer vw.deinit();
    var enc = vae_mod.Encoder.load(gpa, &vw, s) catch return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: VAE load failed");
    defer enc.deinit();

    var lat_parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (lat_parts.items) |x| _ = mlx.mlx_array_free(x);
        lat_parts.deinit(gpa);
    }

    if (stills) {
        for (refs.items) |m| {
            if (m.ref.kind != .image) continue;
            const px = m.pixels orelse return req.refuseFmt(.unsupported, "internal: missing reference pixels", "internal: missing reference pixels", .{});
            const z = enc.encodeImage(px, gpa, s) catch {
                return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: VAE encode failed");
            };
            _ = mlx.mlx_array_eval(z);
            lat_parts.append(gpa, z) catch {
                _ = mlx.mlx_array_free(z);
                return Verdict.refuse(.busy, "out of memory");
            };
        }
        if (lat_parts.items.len == 0)
            return req.refuseFmt(.unsupported, "'ref_images' must contain at least one valid image", "'ref_images' must contain at least one valid image", .{});
    } else {
        if (refs.items.len != 1 or refs.items[0].ref.kind != .video)
            return req.refuseFmt(.unsupported, "'ref_videos' must contain exactly one reference video", "'ref_videos' must contain exactly one reference video", .{});
        const px = refs.items[0].pixels orelse return req.refuseFmt(.unsupported, "internal: missing reference pixels", "internal: missing reference pixels", .{});
        const z = enc.encodeVideo(px, gpa, s) catch {
            return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: VAE encode failed");
        };
        _ = mlx.mlx_array_eval(z);
        lat_parts.append(gpa, z) catch {
            _ = mlx.mlx_array_free(z);
            return Verdict.refuse(.busy, "out of memory");
        };
    }

    const latent: mlx.mlx_array = if (lat_parts.items.len == 1)
        lat_parts.pop().?
    else
        h3_refmod.stackLatentsAlongT(gpa, lat_parts.items, s) catch {
            return job.failure(error.OutOfMemory, "MiniMax-H3 refmods: latent stack failed");
        };
    errdefer _ = mlx.mlx_array_free(latent);
    for (lat_parts.items) |x| _ = mlx.mlx_array_free(x);
    lat_parts.clearRetainingCapacity();

    var meta = h3_refmod.metaFromLatentShape(width, height, latent) catch {
        _ = mlx.mlx_array_free(latent);
        return Verdict.refuse(.failed, "bad visual latent shape");
    };
    if (stills and refs.items.len > 1) meta.source_frames = @intCast(refs.items.len);

    h3_refmod.save(gpa, out_path, meta, latent) catch |err| {
        _ = mlx.mlx_array_free(latent);
        return switch (err) {
            error.OutOfMemory => Verdict.refuse(.busy, "out of memory"),
            else => Verdict.named(.failed, @errorName(err)),
        };
    };
    _ = mlx.mlx_array_free(latent);

    const kind_str: []const u8 = if (meta.kind == .video) "video" else "image";
    e.refmods_result = refmodsOkJson(gpa, out_path, kind_str, meta.latent_t) catch return Verdict.refuse(.busy, "out of memory");
    return .ok;
}

/// `POST /v1/video/refmods` — VAE-encode references to a RefMod `.safetensors` (host inference thread). On success,
/// `refmodsResultTake` returns the JSON body; on refusal, `req.message` may hold the composed text.
pub fn refmodsExtract(ep: *anyopaque, req: *const abi.Request) abi.Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    return runRefmodsExtract(e, req);
}

/// Moves the last success JSON out of `e` (caller frees with `std.heap.c_allocator`).
pub fn refmodsResultTake(ep: *anyopaque) []u8 {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const r = e.refmods_result;
    e.refmods_result = &.{};
    return r;
}

/// A tiny random H3: a one-layer DiT at 64 (two 32-wide heads, one refiner block), a one-layer text encoder at 64
/// over a byte tokenizer, a one-block video VAE decoder at 32 and no audio VAE (the clip comes back silent).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const H = 64;
    const heads = 2;
    const hd = 32;
    const ffn = 64;
    const ted = 32; // time_embed_dim
    const vocab = 260;
    var dit = sdk.tensors.Tensors.init(a, 101);
    try dit.rand("video_patch_proj.weight", .{}, &.{ H, 96 }, .F32, 0.1);
    try dit.rand("video_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
    try dit.rand("audio_patch_proj.weight", .{}, &.{ H, 32 }, .F32, 0.1);
    try dit.rand("audio_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
    try dit.rand("condition_proj.weight", .{}, &.{ H, H }, .BF16, 0.1);
    try dit.rand("condition_proj.bias", .{}, &.{H}, .BF16, 0.1);
    try dit.rand("time_embedder.proj_in.weight", .{}, &.{ H, 32 }, .F32, 0.1);
    try dit.rand("time_embedder.proj_in.bias", .{}, &.{H}, .F32, 0.1);
    try dit.rand("time_embedder.proj_out.weight", .{}, &.{ ted, H }, .F32, 0.1);
    try dit.rand("time_embedder.proj_out.bias", .{}, &.{ted}, .F32, 0.1);
    try dit.values("rope.inv_freq", .{}, &.{4}, .F32, &.{ 1.0, 0.3, 0.1, 0.03 });
    inline for (.{ "blocks.0", "token_refiner.blocks.0" }) |p| {
        try dit.rand(p ++ ".attn.qkv_proj.weight", .{}, &.{ 3 * heads * hd, H }, .BF16, 0.1);
        try dit.rand(p ++ ".attn.out_proj.weight", .{}, &.{ H, heads * hd }, .BF16, 0.1);
        try dit.fill(p ++ ".attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
        try dit.fill(p ++ ".attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);
        try dit.rand(p ++ ".mlp.fc1.weight", .{}, &.{ 2 * ffn, H }, .BF16, 0.1);
        try dit.rand(p ++ ".mlp.fc2.weight", .{}, &.{ H, ffn }, .BF16, 0.1);
        try dit.fill(p ++ ".norm1.weight", .{}, &.{H}, .BF16, 1);
        try dit.fill(p ++ ".norm2.weight", .{}, &.{H}, .BF16, 1);
    }
    try dit.rand("blocks.0.adaln_proj.linear.weight", .{}, &.{ 18 * H, ted }, .BF16, 0.05);
    try dit.rand("blocks.0.adaln_proj.linear.bias", .{}, &.{18 * H}, .BF16, 0.05);
    try dit.fill("token_refiner.final_norm.weight", .{}, &.{H}, .BF16, 1);
    try dit.rand("final_layer.adaln_proj.linear.weight", .{}, &.{ 2 * H, ted }, .BF16, 0.05);
    try dit.rand("final_layer.adaln_proj.linear.bias", .{}, &.{2 * H}, .BF16, 0.05);
    try dit.fill("final_layer.norm.weight", .{}, &.{H}, .BF16, 1);
    try dit.rand("final_layer.video_out.weight", .{}, &.{ 96, H }, .F32, 0.1);
    try dit.rand("final_layer.video_out.bias", .{}, &.{96}, .F32, 0.1);
    try dit.rand("final_layer.audio_out.weight", .{}, &.{ 32, H }, .F32, 0.1);
    try dit.rand("final_layer.audio_out.bias", .{}, &.{32}, .F32, 0.1);

    var te = sdk.tensors.Tensors.init(a, 102);
    try te.rand("model.embed_tokens.weight", .{}, &.{ vocab, H }, .BF16, 0.5);
    const tp = "model.layers.0.";
    try te.fill(tp ++ "input_layernorm.weight", .{}, &.{H}, .BF16, 1);
    try te.fill(tp ++ "post_attention_layernorm.weight", .{}, &.{H}, .BF16, 1);
    inline for (.{ .{ "self_attn.q_proj", 2 * hd, H }, .{ "self_attn.k_proj", hd, H }, .{ "self_attn.v_proj", hd, H }, .{ "self_attn.o_proj", H, 2 * hd }, .{ "mlp.gate_proj", 128, H }, .{ "mlp.up_proj", 128, H }, .{ "mlp.down_proj", H, 128 } }) |l|
        try te.rand(tp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.1);
    try te.fill(tp ++ "self_attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
    try te.fill(tp ++ "self_attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);

    const vd = 32; // decoder width: 2 heads x 16
    var vae = sdk.tensors.Tensors.init(a, 103);
    try vae.rand("decoder.x_embedder.weight", .{}, &.{ vd, 24 }, .BF16, 0.2);
    try vae.rand("decoder.x_embedder.bias", .{}, &.{vd}, .BF16, 0.1);
    try vae.rand("decoder.register_tokens", .{}, &.{ 1, 4, vd }, .BF16, 0.2);
    try vae.fill("decoder.norm_out.weight", .{}, &.{vd}, .BF16, 1);
    try vae.fill("decoder.norm_out.bias", .{}, &.{vd}, .BF16, 0);
    try vae.rand("decoder.proj_out.weight", .{}, &.{ 3072, vd }, .BF16, 0.2);
    try vae.rand("decoder.proj_out.bias", .{}, &.{3072}, .BF16, 0.1);
    try vae.rand("decoder.mask_token", .{}, &.{ 1, 1, vd }, .BF16, 0.2);
    try vae.rand("post_quant_conv.weight", .{}, &.{ 24, 24, 1, 1, 1 }, .BF16, 0.2);
    try vae.rand("post_quant_conv.bias", .{}, &.{24}, .BF16, 0.1);
    try vae.fill("latents_mean", .{}, &.{24}, .BF16, 0);
    try vae.fill("latents_std", .{}, &.{24}, .BF16, 1);
    const vp = "decoder.transformer_blocks.0.";
    inline for (.{ .{ "attn.to_qkv", 3 * vd, vd }, .{ "attn.to_out", vd, vd }, .{ "ff.w1", 8 * vd, vd }, .{ "ff.w2", vd, 4 * vd } }) |l| {
        try vae.rand(vp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.2);
        try vae.rand(vp ++ l[0] ++ ".bias", .{}, &.{l[1]}, .BF16, 0.1);
    }
    inline for (.{ "norm1.weight", "norm2.weight", "scale1", "scale2" }) |n| try vae.fill(vp ++ n, .{}, &.{vd}, .BF16, 1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = 
            \\{"model_type":"minimax_h3","partition":"fl2va","tasks":["t2va","fl2va"],
            \\ "transformer":{"hidden_size":64,"num_layers":1,"token_refiner_num_layers":1,"num_attention_heads":2,
            \\  "attention_head_dim":32,"ffn_hidden_size":64,"text_dim":64,"timestep_input_dim":32,
            \\  "time_embed_hidden_size":64,"time_embed_dim":32,"rope_inv_freq_len":4},
            \\ "text_encoder":{"hidden":64,"layers":1,"heads":2,"kv_heads":1,"head_dim":32,"intermediate":128},
            \\ "video_vae":{"layers":1,"heads":2,"head_dim":16}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{}) },
            .{ .name = "transformer.safetensors", .bytes = try dit.bytes() },
            .{ .name = "text_encoder.safetensors", .bytes = try te.bytes() },
            .{ .name = "video_vae.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red ball rolls\",\"width\":64,\"height\":64,\"num_frames\":5,\"steps\":2,\"seed\":3,\"fast\":false}",
    };
}

/// Same weights as `conformanceFixture` but advertises `ref2va` for hybrid gate tests.
pub fn hybridConformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = try conformanceFixture(a);
    const files = try a.alloc(sdk.fixture.File, fx.files.len);
    for (fx.files, files) |file, *dst| {
        if (std.mem.eql(u8, file.name, "config.json")) {
            dst.* = .{
                .name = file.name,
                .bytes = try std.mem.replaceOwned(u8, a, file.bytes, "\"tasks\":[\"t2va\",\"fl2va\"]", "\"tasks\":[\"t2va\",\"fl2va\",\"ref2va\"]"),
            };
        } else {
            dst.* = file;
        }
    }
    return .{ .files = files, .body = fx.body };
}

// ── tests ──

const testing = std.testing;

test "minimax_h3 plugin: a seeded clip on the tiny pack is deterministic JSON members; bad fields are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expect(std.mem.startsWith(u8, first, "\"frames\":5,\"height\":64,\"width\":64,\"fps\":24,\"format\":\"rgb8\",\"data\":\""));
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    inline for (.{
        .{ "{}", "missing 'prompt'" },
        .{ "{\"prompt\":\"x\",\"width\":70}", "width and height must be multiples of 32" },
        .{ "{\"prompt\":\"x\",\"steps\":0}", "'steps' must be in [1,100]" },
        .{ "{\"prompt\":\"x\",\"chain_windows\":7}", "chain_windows must be 1-6" },
        .{ "{\"prompt\":\"x\",\"latent_prefix_frames\":22,\"chain_windows\":1}", "latent_prefix_frames requires chain_windows > 1" },
        .{ "{\"prompt\":\"x\",\"latent_prefix_path\":\"/tmp/no_h3_sidecar.safetensors\"}", "latent_prefix_path requires latent_prefix_frames" },
        .{ "{\"prompt\":\"x\",\"latent_out\":\"latents.safetensors\"}", "'latent_out' must be an absolute path" },
        .{ "{\"prompt\":\"x\",\"refine_sigmas\":[0.9,0]}", "latent_init_path is required for refine / latent_upscale" },
        .{ "{\"prompt\":\"x\",\"latent_upscale\":{\"factor\":2}}", "latent_init_path is required for refine / latent_upscale" },
        .{ "{\"prompt\":\"x\",\"latent_init_path\":\"latents.safetensors\"}", "'latent_init_path' must be an absolute path" },
        .{ "{\"prompt\":\"x\",\"latent_init_path\":\"/tmp/no_h3_init.safetensors\",\"chain_windows\":2}", "latent refine does not support chain_windows > 1" },
        .{ "{\"prompt\":\"x\",\"latent_init_path\":\"/tmp/no_h3_init.safetensors\",\"refine_sigmas\":[0.5,0.9,0]}", "refine_sigmas must be strictly descending" },
        .{ "{\"prompt\":\"x\",\"ref_images\":[\"AAAA\"]}", "this MiniMax-H3 pack does not support references (it declares no 'ref2va' task) — load a REF2VA checkpoint" },
        .{ "{\"prompt\":\"x\",\"turbo\":true}", "this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors" },
        .{ "{\"prompt\":\"x\",\"first_frame_image\":\"!!\"}", "keyframe image is not valid base64" },
    }) |c| try testing.expectEqualStrings(c[1], (try pack.run(c[0])).refused.why.slice());
}

test "minimax_h3 plugin: a canvas past the live memory is refused before any stage loads, naming its numbers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const m: abi.Machine = .{ .ram_total = 1 << 30, .ram_available = 1 << 20 };
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    const dir = pack.mem.packDir();
    const req: abi.Request = .{ .body = .of(pack.fx.body), .machine = &m, .dir = &dir, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const Quiet = struct {
        fn report(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}
        fn boundary(_: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    };
    const v = table.step(pack.engine, &req, &.{ .ctx = null, .report = Quiet.report, .boundary = Quiet.boundary });
    try testing.expect(v.code == .unsupported);
    try testing.expect(std.mem.startsWith(u8, buf[0..len], "64x64 x 5 frames needs ~"));
    try testing.expect(std.mem.endsWith(u8, buf[0..len], "GB is available: use a smaller canvas or fewer frames"));
}

test "minimax_h3 plugin: a reference the server cannot use is refused by name, never dropped" {
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    const m: abi.Machine = .{};
    const req: abi.Request = .{ .body = .of("{}"), .machine = &m, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cases = [_]struct { body: []const u8, needle: []const u8 }{
        .{ .body = "{\"ref_images\":[\"QUJD\",\"!!!!\"]}", .needle = "'ref_images'[1]" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",\"QUJD\",\"!!!!\"]}]}", .needle = "frames[2]" },
        .{ .body = "{\"ref_audios\":[\"!!!!\"]}", .needle = "'ref_audios'[0]" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\"],\"audio\":\"QUJD\"}]}", .needle = "'ref_videos'[0].audio" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",1]}]}", .needle = "'ref_videos'[0].frames" },
        .{ .body = "{\"ref_images\":[\"QUJD\"],\"ref_image_size\":\"huge\"}", .needle = "ref_image_size" },
        .{ .body = "{\"ref_images\":[\"QUJDRA==\"]}", .needle = "could not decode 'ref_images'[0]" },
    };
    for (cases) |c| {
        var b = job.Body.parse(testing.allocator, c.body).?;
        defer b.deinit();
        var out: std.ArrayList(h3.RefMedia) = .empty;
        defer {
            for (out.items) |*r| r.deinit();
            out.deinit(gpa);
        }
        len = 0;
        const v = (try references(&req, &b, 864, 480, 124, s, &out, h3.RefLimits.generation, 0)) orelse return error.ExpectedRejection;
        const msg = if (len > 0) buf[0..len] else v.why.slice();
        testing.expect(std.mem.indexOf(u8, msg, c.needle) != null) catch |e| {
            std.debug.print("{s}: {s}\n", .{ c.body, msg });
            return e;
        };
    }
    // No reference fields at all is not a rejection: the feature is simply off.
    var none_b = job.Body.parse(testing.allocator, "{\"prompt\":\"x\"}").?;
    defer none_b.deinit();
    var none: std.ArrayList(h3.RefMedia) = .empty;
    defer none.deinit(gpa);
    try testing.expect((try references(&req, &none_b, 864, 480, 124, s, &none, h3.RefLimits.generation, 0)) == null);
    try testing.expectEqual(@as(usize, 0), none.items.len);
}

test "minimax_h3 plugin: reference audio arrives as 32 kHz stereo in the encoder's planar shape" {
    // The audio VAE is mono and takes the stereo channels on the batch axis, so interleaved -> [2, L] is a
    // de-interleave, not a reshape (a reshape runs and encodes a channel-swapped chirp).
    const inter = [_]f32{ 0.0, 0.5, 0.1, 0.6, 0.2, 0.7 };
    const arr = try planar(&inter);
    defer _ = mlx.mlx_array_free(arr);
    try testing.expectEqualSlices(c_int, &.{ 2, 3 }, mlx.getShape(arr));
    try mlx.check(mlx.mlx_array_eval(arr));
    const d = mlx.mlx_array_data_float32(arr).?;
    for ([_]f32{ 0.0, 0.1, 0.2, 0.5, 0.6, 0.7 }, 0..) |v, i| try testing.expectApproxEqAbs(v, d[i], 1e-6);
    // A 16 kHz mono clip comes back at the VAE's 32 kHz in stereo: the latent-frame count is computed from the
    // rate, so a clip left at its own rate halves the reference's length.
    var mono: [1600]f32 = undefined;
    for (&mono, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    const wav_bytes = try wav.encodePcm16(testing.allocator, &mono, 16000, 1);
    defer testing.allocator.free(wav_bytes);
    const b64 = try testing.allocator.alloc(u8, std.base64.standard.Encoder.calcSize(wav_bytes.len));
    defer testing.allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, wav_bytes);
    const pcm = try wav32kStereo(b64);
    defer gpa.free(pcm);
    try testing.expectEqual(@as(usize, 3200 * 2), pcm.len);
    // Duplicated, not summed: both channels carry the mono signal.
    try testing.expectApproxEqAbs(pcm[0], pcm[1], 1e-6);
    try testing.expectApproxEqAbs(pcm[200], pcm[201], 1e-6);
}

test "minimax_h3 plugin: refmods canvas gate skips width/height on voice extract" {
    try testing.expect(refmodsCanvasError(true, 0, 0) == null);
    try testing.expectEqualStrings("missing 'width' or 'height'", refmodsCanvasError(false, 0, 0).?);
    try testing.expectEqualStrings("width and height must be multiples of 32", refmodsCanvasError(false, 70, 64).?);
    try testing.expect(refmodsCanvasError(false, 256, 256) == null);
}

/// The tiny pack's config and tokenizer plus a 32-wide LatentResizer2D (one in and one out block) and no DiT, text
/// encoder or VAE files: an upscale-only job that reached any of them fails.
fn upscaleOnlyFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const C = 32;
    var w = sdk.tensors.Tensors.init(a, 104);
    try w.rand("resizer.conv_in.weight", .{}, &.{ C, 24, 3, 3 }, .F32, 0.05);
    try w.rand("resizer.conv_in.bias", .{}, &.{C}, .F32, 0.05);
    try w.rand("resizer.embed.linear.0.weight", .{}, &.{ C, 96 }, .F32, 0.05);
    try w.rand("resizer.embed.linear.2.weight", .{}, &.{ C, C }, .F32, 0.05);
    inline for (.{ "resizer.in_blocks.0", "resizer.out_blocks.0" }) |p| {
        inline for (.{ ".in_layers.0", ".out_norm" }) |n| {
            try w.fill(p ++ n ++ ".weight", .{}, &.{C}, .F32, 1);
            try w.fill(p ++ n ++ ".bias", .{}, &.{C}, .F32, 0);
        }
        try w.rand(p ++ ".in_layers.2.weight", .{}, &.{ C, C, 3, 3 }, .F32, 0.05);
        try w.rand(p ++ ".out_layers.2.weight", .{}, &.{ C, C, 3, 3 }, .F32, 0.05);
        try w.rand(p ++ ".emb_layers.1.weight", .{}, &.{ 2 * C, C }, .F32, 0.05);
    }
    try w.fill("resizer.norm_out.weight", .{}, &.{C}, .F32, 1);
    try w.fill("resizer.norm_out.bias", .{}, &.{C}, .F32, 0);
    try w.rand("resizer.conv_out.weight", .{}, &.{ 24, C, 3, 3 }, .F32, 0.05);
    const fx = try conformanceFixture(a);
    var files: std.ArrayList(sdk.fixture.File) = .empty;
    for (fx.files) |f| if (!std.mem.endsWith(u8, f.name, ".safetensors")) try files.append(a, f);
    try files.append(a, .{ .name = "latent_upscaler/tiny.safetensors", .bytes = try w.bytes() });
    return .{ .files = try files.toOwnedSlice(a), .body = fx.body };
}

/// A deterministic `[shape]` f32 ramp (owned).
fn rampArray(shape: []const c_int, scale: f32) !A {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = scale * @as(f32, @floatFromInt(@as(i64, @intCast(i % 97)) - 48)) / 48.0;
    return mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
}

fn expectSameArray(want: A, got: A) !void {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

test "minimax_h3 plugin: refine:false upscales any latent T to latent_out, carries the audio, and never needs the DiT" {
    const up_table = table;
    const UpOnly = struct {
        pub const plugin = config.plugin;
        pub const table = up_table;
        pub const conformanceFixture = upscaleOnlyFixture;
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var pack = try @import("../../sdk/testing/media_pack.zig").MediaPack(UpOnly).open(aa);
    defer pack.close();
    const root_buf = try aa.alloc(u8, std.fs.max_path_bytes);
    const root = root_buf[0..try pack.tmp.dir.realPath(testing.io, root_buf)];
    const init_path = try std.fmt.allocPrintSentinel(aa, "{s}/run.safetensors", .{root}, 0);
    const out_path = try std.fmt.allocPrintSentinel(aa, "{s}/run_up.safetensors", .{root}, 0);

    // 40 latent frames: off the 17k+5 grid and past one upscaler chunk.
    const v = try rampArray(&.{ 1, 24, 40, 4, 4 }, 1.0);
    defer _ = mlx.mlx_array_free(v);
    const au = try rampArray(&.{ 1, 32, 2, 61 }, 0.5);
    defer _ = mlx.mlx_array_free(au);
    try h3.saveLatentSidecar(init_path.ptr, v, au, pack.stream);

    const body = try std.fmt.allocPrint(aa, "{{\"refine\":false,\"latent_init_path\":\"{s}\",\"latent_upscale\":{{\"factor\":2,\"tier\":1}},\"latent_out\":\"{s}\"}}", .{ init_path, out_path });
    const got = (try pack.run(body)).ok;
    const want = try std.fmt.allocPrint(aa, "\"latent_out\":\"{s}\",\"latent_t\":40,\"width\":128,\"height\":128", .{out_path});
    try testing.expectEqualStrings(want, got);
    const out = try h3.loadLatentSidecar(aa, out_path.ptr, pack.stream);
    defer _ = mlx.mlx_array_free(out.video);
    defer _ = mlx.mlx_array_free(out.audio);
    try testing.expectEqualSlices(c_int, &.{ 1, 24, 40, 8, 8 }, mlx.getShape(out.video));
    try expectSameArray(au, out.audio);

    const base = try std.fmt.allocPrint(aa, "\"refine\":false,\"latent_init_path\":\"{s}\",\"latent_upscale\":{{\"factor\":2}}", .{init_path});
    inline for (.{
        .{ "", "refine:false (upscale only) needs latent_out" },
        .{ ",\"latent_out\":\"/tmp/x.safetensors\",\"refine_sigmas\":[0.5,0]", "refine:false (upscale only) takes no refine_sigmas, refine_start_sigma or refine_audio" },
        .{ ",\"latent_out\":\"/tmp/x.safetensors\",\"latent_prefix_frames\":5", "refine:false (upscale only) takes no pins, references, keyframes or chain_windows" },
        .{ ",\"latent_out\":\"/tmp/x.safetensors\",\"width\":96,\"height\":128", "'latent_upscale.factor' does not match request width/height" },
    }) |c| {
        const bad_body = try std.fmt.allocPrint(aa, "{{{s}{s}}}", .{ base, c[0] });
        try testing.expectEqualStrings(c[1], (try pack.run(bad_body)).refused.why.slice());
    }
    try testing.expectEqualStrings("'refine' must be true or false", (try pack.run("{\"refine\":0}")).refused.why.slice());
}

const tiny_ref_png_b64 ="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mMQEREBAAB8AD2sffR3AAAAAElFTkSuQmCC";

test "minimax_h3 plugin: ref compat and latent_out parent refuse before reference decode" {
    const hybrid_table = table;
    const hybrid_plugin = plugin;
    const Hybrid = struct {
        pub const plugin = hybrid_plugin;
        pub const table = hybrid_table;
        pub fn conformanceFixture(alloc: std.mem.Allocator) !sdk.fixture.Fixture {
            return hybridConformanceFixture(alloc);
        }
    };
    const HybridPack = @import("../../sdk/testing/media_pack.zig").MediaPack(Hybrid);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try HybridPack.open(arena.allocator());
    defer pack.close();
    const chain_body = try std.fmt.allocPrint(arena.allocator(), "{{\"prompt\":\"x\",\"num_frames\":56,\"width\":64,\"height\":64,\"ref_images\":[\"{s}\"],\"chain_windows\":2}}", .{tiny_ref_png_b64});
    const latout_body = try std.fmt.allocPrint(arena.allocator(), "{{\"prompt\":\"x\",\"num_frames\":56,\"width\":64,\"height\":64,\"ref_images\":[\"{s}\"],\"latent_out\":\"/tmp/no_such_parent_dir_zz/latents.safetensors\"}}", .{tiny_ref_png_b64});
    inline for (.{
        .{ chain_body, "chained windows cannot be used with references or ref_mod_paths" },
        .{ latout_body, "'latent_out' parent directory does not exist" },
    }) |c| {
        const out = try pack.run(c[0]);
        try testing.expectEqualStrings(c[1], out.refused.why.slice());
    }
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
