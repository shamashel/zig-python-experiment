//! This module exposes a Byte-Pair Encoding Tokenizer.
//! This should support any Byte-Pair Encoder's vocab and merges files.
const std = @import("std");

// Support unreasonably large vocab / merges files because I'm paranoid
const MAX_VOCAB_BYTES = 32 * 1024 * 1024;
const MAX_MERGES_BYTES = 8 * 1024 * 1024;

pub const UTF8EncodedChar = u8;
pub const UTF8EncodedStr = []const UTF8EncodedChar;
pub const UTF8EncodedStrPtr = [*:0]const UTF8EncodedChar; // C-Style str, null terminated
pub const UTF8EncodedBytesPtr = [*]const UTF8EncodedChar; // Raw bytes, assumes we know the length

pub const TokenId = u32;
pub const OutTokenIds = []TokenId;
pub const OutTokenIdsPtr = [*]TokenId;

pub const VocabEntry = struct {
    bytes: UTF8EncodedStr, // The JSON Key
    id: TokenId, // The JSON Value
};
const VocabMap = std.json.ArrayHashMap(TokenId);

pub const MergeRule = struct { left: TokenId, right: TokenId };
const RuleList = std.ArrayList(MergeRule);

pub const EncoderMetadata = struct { vocab: []const VocabEntry, merges: []const MergeRule };
pub const EncoderContext = struct {
    allocator: std.heap.ArenaAllocator,
    meta: EncoderMetadata,
};

pub const BlackMagic = ?*anyopaque;

fn read_vocab_file(allocator: std.mem.Allocator, vocab_path: UTF8EncodedStr) !std.json.Parsed(VocabMap) {
    // Read file
    const cwd = std.fs.cwd();
    const vocab_file = try cwd.readFileAlloc(allocator, vocab_path, MAX_VOCAB_BYTES);
    defer allocator.free(vocab_file);

    // Parse as JSON. Apparently the default is .allocate = .alloc_if_needed, which causes a segfault
    return try std.json.parseFromSlice(VocabMap, allocator, vocab_file, .{ .allocate = .alloc_always });
}

/// Parse the provided `vocab.json` file into an array of VocabEntry
fn parse_vocab(allocator: std.mem.Allocator, vocab_map: VocabMap) ![]VocabEntry {
    // Convert to []VocabEntry
    var entries = try allocator.alloc(VocabEntry, vocab_map.map.count());
    errdefer allocator.free(entries);
    var iterator = vocab_map.map.iterator();
    var i: usize = 0; // usize since max length is the max array size
    while (iterator.next()) |entry| {
        entries[i] = VocabEntry{
            // Need to copy these so they don't get freed when `parsed` gets dealloc'd
            .bytes = try allocator.dupe(UTF8EncodedChar, entry.key_ptr.*),
            .id = @intCast(entry.value_ptr.*),
        };
        i += 1;
    }
    return entries;
}

fn parse_merge_rules(allocator: std.mem.Allocator, merges_path: UTF8EncodedStr, vocab_map: VocabMap) ![]MergeRule {
    // Read file
    const cwd = std.fs.cwd();
    const merges_file = try cwd.readFileAlloc(allocator, merges_path, MAX_MERGES_BYTES);
    defer allocator.free(merges_file);

    var lines = std.mem.tokenizeScalar(UTF8EncodedChar, merges_file, '\n');
    // Format is supposed to be a comment on line 1, followed by space-delimited left-right mappings
    _ = lines.next() orelse return error.EmptyMergesFile;

    var rule_list: RuleList = .{};
    errdefer rule_list.deinit(allocator);

    while (lines.next()) |raw_line| {
        // Trim whitespace
        const trimmed = std.mem.trim(UTF8EncodedChar, raw_line, "\t\r");
        if (trimmed.len == 0) continue;

        // Technically you can split by either a space or a tab, so tokenize by both just in case
        // NOTE: We're using `tokenize*` instead of `split*` here since we technically don't care about delimiting by multiple spaces
        var tokenized_line = std.mem.tokenizeAny(UTF8EncodedChar, trimmed, " \t");
        // I think we're supposed to ignore incomplete pairs instead of error out on them? Can change this later if otherwise.
        const left_str = tokenized_line.next() orelse continue;
        const right_str = tokenized_line.next() orelse continue;

        const left = vocab_map.map.get(left_str) orelse {
            std.debug.print("Unknown left token: `{s}`", .{left_str});
            return error.UnknownLeftToken;
        };
        const right = vocab_map.map.get(right_str) orelse {
            std.debug.print("Unknown right token: `{s}`", .{right_str});
            return error.UnknownRightToken;
        };

        try rule_list.append(allocator, MergeRule{
            .left = left,
            .right = right,
        });
    }

    // Converts from ArrayList to slice and handles the `rule_list.deinit()` call for us
    return rule_list.toOwnedSlice(allocator);
}

/// Prepare the encoder for tokenization by loading the trained vocab and merges
pub fn internal_prepare_encoder(allocator: std.mem.Allocator, vocab_path: UTF8EncodedStr, merges_path: UTF8EncodedStr) !EncoderMetadata {
    const vocab_map_parsed = try read_vocab_file(allocator, vocab_path);
    // Also handles the dealloc for underlying `vocab_map`
    defer vocab_map_parsed.deinit();
    const vocab_map = vocab_map_parsed.value;

    // Convert map to the final list of VocabEntry, separate alloc from vocab_map
    const vocab = try parse_vocab(allocator, vocab_map);
    errdefer allocator.free(vocab);

    // Parse the merge rules and convert each rule from strings to tokens, separate alloc from vocab_map
    const merge_rules = try parse_merge_rules(allocator, merges_path, vocab_map);
    errdefer allocator.free(merge_rules);
    return EncoderMetadata{
        .vocab = vocab,
        .merges = merge_rules,
    };
}

/// Initialize the encoder with the given vocab.
/// Wraps the `internal_prepare_encoder` method with an ABI-friendly interface.
///
/// Returns a pointer to the EncoderContext on success and null otherwise.
export fn init(vocab_path: UTF8EncodedStrPtr, merges_path: UTF8EncodedStrPtr) BlackMagic {
    const config = std.heap.page_allocator.create(EncoderContext) catch |err| {
        std.debug.print("Error while creating EncoderContext: {any}\n", .{err});
        return null;
    };
    config.allocator = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = config.allocator.allocator();

    // Also not sure where this should live/how this should work?
    const meta = internal_prepare_encoder(
        alloc,
        std.mem.span(vocab_path),
        std.mem.span(merges_path),
    ) catch |err| {
        std.debug.print("Error while creating encoder: {any}\n", .{err});
        // Nuke the arena if we fail to create the metadata
        config.allocator.deinit();
        std.heap.page_allocator.destroy(config);
        return null;
    };
    config.meta = meta;
    return config;
}

export fn deinit(context: BlackMagic) c_int {
    // How do we get this here?
    const real_context: *EncoderContext = @ptrCast(@alignCast(context));
    real_context.allocator.deinit();
    std.heap.page_allocator.destroy(real_context);
    return 0;
}

// /// Encoder method for the tokenizer.
// ///
// /// Returns a slice of tokens
// pub fn internal_encode(allocator: std.mem.Allocator, text: UTF8EncodedStr, meta: *const EncoderMetadata) !OutTokenIds {}

// /// Encode the text based on the initialized BPE tokenizer.
// /// Wraps the `internal_encode` method with an ABI-friendly interface.
// ///
// /// Returns:
// /// - [>=0]: The number of tokens written (slice from 0 to this number for an accurate list of tokens)
// /// - [-1]: Generic error, check logs
// /// - [-2]: Capacity too small. Ensure the output ids array has the same length as the `text_str`.
// export fn encode(text_str: UTF8EncodedBytesPtr, text_len: c_int, out_ids_ptr: OutTokenIdsPtr, out_capacity: c_int) c_int {
//     // Don't need to tokenize if there's nothing to tokenize
//     if (text_len == 0) return 0;
//     if (text_len < 0 or out_capacity < 0) {
//         std.debug.print(
//             "ERROR: Text length or output capacity less than 0. \nText Length: {}\nOutput Capacity: {}\n",
//             .{ text_len, out_capacity },
//         );
//         return -1;
//     }
//     if (out_capacity < text_len) {
//         std.debug.print(
//             "ERROR: Output capacity should be at least as large as the number of input bytes. \nText Length: {}\nOutput Capacity: {}\n",
//             .{ text_len, out_capacity },
//         );
//         return -2;
//     }
// }

test "init loads vocab and merges" {
    // Adjust filenames if yours differ
    const vocab_path: UTF8EncodedStrPtr = "vocab.json";
    const merges_path: UTF8EncodedStrPtr = "merges.txt";

    const context = init(vocab_path, merges_path);
    const real_context: *EncoderContext = @ptrCast(@alignCast(context));
    const meta = real_context.meta;
    defer _ = deinit(context);

    std.testing.expect(meta.vocab.len > 0) catch |err| {
        std.debug.print("vocab len was 0, error: {s}\n", .{@errorName(err)});
        return err;
    };

    std.testing.expect(meta.merges.len > 0) catch |err| {
        std.debug.print("merges len was 0, error: {s}\n", .{@errorName(err)});
        return err;
    };

    std.debug.print("Loaded vocab={d} merges={d}\n", .{ meta.vocab.len, meta.merges.len });
}
