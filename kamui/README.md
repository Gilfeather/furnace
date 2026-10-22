# kamui

Named after Kamui (神威), the power to put things away in another space and bring them back, as
kamui keeps freed memory and hands it out again.
Built for and maintained in [Furnace](https://github.com/Gilfeather/furnace), the pure-Rust
inference server.

A global allocator that keeps freed large blocks for reuse, with no dependencies.

Every operator of a neural network allocates its output, and a forward pass makes dozens of
allocations between 100 KB and a few MB. The system allocators hand blocks that large straight
back to the OS on free, so the next pass pays for fresh pages again: a page fault per 4 KiB on
first touch, plus zeroing. `BlockCache` keeps up to 128 MiB of freed blocks of 256 KiB or more
and hands them out again to allocations of the same size and alignment, which a repeated
forward pass makes in the same order. Smaller allocations go straight to the system allocator.

```toml
[dependencies]
kamui = "0.15"
```

```rust
#[global_allocator]
static ALLOCATOR: kamui::BlockCache = kamui::BlockCache::new();
```

The `furnace` command of [Furnace](https://github.com/Gilfeather/furnace) runs on it. The API
documentation is on [docs.rs](https://docs.rs/kamui). MIT licensed.
