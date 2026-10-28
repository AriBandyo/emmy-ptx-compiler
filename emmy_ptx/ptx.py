def build_ptx(
    *, kernel_meta, arg_order, tile, axis,
    use_dynamic_shared, pool_alignment,
    shared_declarations, instructions,
    pointer_instructions, body_instructions,
    value_regs,
):
    ptx_lines = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
        f".visible .entry {kernel_meta['entry']}("
    ]

    for j, arg_name in enumerate(arg_order):
        comma = "," if j < len(arg_order) - 1 else ""

        ptx_lines.append(
            f"    .param .u64 param_{arg_name}{comma}"
        )

    ptx_lines.append(")")
    ptx_lines = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
        f".visible .entry {kernel_meta['entry']}("
    ]

    for j, arg_name in enumerate(arg_order):
        comma = "," if j < len(arg_order) - 1 else ""
        ptx_lines.append(
            f"    .param .u64 param_{arg_name}{comma}"
        )

    ptx_lines.append(")")




    if use_dynamic_shared:
        ptx_lines.insert(
            4,
            f".extern .shared .align {pool_alignment} .b8 _smem_pool[];"
        )

    ptx_lines.append("{")

    if not use_dynamic_shared:
        for declaration in dict.fromkeys(shared_declarations):
            ptx_lines.append(f"    {declaration}")

    ptx_lines.append(
        "    .reg .u32 %r_tid, %r_bid, %r_bdim, %r_a0;"
    )
    ptx_lines.append("    .reg .pred %p_out;")

    # Registers for warp and lane identification
    ptx_lines.append(
        "    .reg .u32 %r_lane, %r_warp;"
    )

    # Registers for shared-memory addressing
    ptx_lines.append(
        "    .reg .u32 %r_smem_base, %r_smem_off, %r_smem_addr;"
    )


    ptx_lines.append(
        "    .reg .pred %p_not_lane0;"
    )
    ptx_lines.append("    .reg .pred %p_not_tree0;")
    ptx_lines.append("    .reg .pred %p_not_output;")

    ptx_lines.append("    .reg .pred %p_loop;")
    ptx_lines.append("    .reg .pred %p_inner_loop;")
    ptx_lines.append("    .reg .pred %p_async_loop;")
    ptx_lines.append("    .reg .pred %p_ternary;")
    if any(type(op).__name__ == "WarpShuffle" for op in tile.body):
        ptx_lines.append(
            "    .reg .b32 %r_shfl_src, %r_shfl_dst;"
        )
    def collect_loop_axes(nodes):
        for node in nodes:
            if type(node).__name__ in ("Loop", "StridedLoop"):
                yield node.axis.name
                yield from collect_loop_axes(node.body)

    axis_names = {a.name for a in tile.axes}
    axis_names.update(collect_loop_axes(tile.body))
    axis_names.update(
        op.axis.name
        for op in tile.body
        if type(op).__name__ == "StridedLoop"
    )

    axis_names.update(
        inner.axis.name
        for outer in tile.body
        if type(outer).__name__ == "StridedLoop"
        for inner in outer.body
        if type(inner).__name__ == "Loop"
    )
    axis_names.update(
        inner.axis.name
        for outer in tile.body
        if type(outer).__name__ == "StridedLoop"
        for inner in outer.body
        if type(inner).__name__ == "StridedLoop"
    )
    for name in sorted(axis_names):
        if name != "a0":
            ptx_lines.append(f"    .reg .u32 %r_{name};")

    ptx_lines.append(f"    .reg .u32 %r<{len(instructions) + 1}>;")
    ptx_lines.append(f"    .reg .u64 %rd_param<{len(arg_order)}>;")
    ptx_lines.append(f"    .reg .u64 %rd_base<{len(arg_order)}>;")
    if any(type(op).__name__ == "Loop" for op in tile.body):
        ptx_lines.append(
            "    .reg .u64 %rd_regular_offset, %rd_regular_addr;"
        )
    ptx_lines.append(f"    .reg .u64 %rd_offset<{len(tile.body)}>;")
    ptx_lines.append(f"    .reg .u64 %rd_addr<{len(tile.body)}>;")
    ptx_lines.append(f"    .reg .f32 %f<{len(value_regs)}>;")


    if any(
        "cp.async." in inst and ".shared.global" in inst
        for inst in body_instructions
    ):

        ptx_lines.append(
            "    .reg .u32 %r_cp_base, %r_cp_offset, %r_cp_addr;"
        )
        ptx_lines.append(
            "    .reg .u64 %rd_cp_offset, %rd_cp_addr;"
        )
    ptx_lines.extend([
        "    mov.u32 %r_tid, %tid.x;",
        "    mov.u32 %r_bid, %ctaid.x;",
        "    mov.u32 %r_bdim, %ntid.x;"
    ])
    if len(tile.axes) >= 3:
        total = 1

        for a in tile.axes:
            total *= a.extent.as_static()

        ptx_lines.extend([
            "    mad.lo.u32 %r_a0, %r_bid, %r_bdim, %r_tid;",
            f"    setp.ge.u32 %p_out, %r_a0, {total};",
            "    @%p_out bra DONE;"
        ])

        stride = total

        for a in tile.axes:
            extent = a.extent.as_static()
            stride //= extent

            if stride > 1:
                ptx_lines.append(
                    f"    div.u32 %r_{a.name}, %r_a0, {stride};"
                )

                if stride * extent != total:
                    ptx_lines.append(
                        f"    rem.u32 %r_{a.name}, %r_{a.name}, {extent};"
                    )
            else:
                ptx_lines.append(
                    f"    rem.u32 %r_{a.name}, %r_a0, {extent};"
                )

    elif len(tile.axes) == 2:

        # r04: one block processes one matrix row
        ptx_lines.extend([
            "    mov.u32 %r_a0, %r_bid;",
            f"    mov.u32 %r_{tile.axes[1].name}, %r_tid;",
            f"    setp.ge.u32 %p_out, %r_a0, {axis.extent.as_static()};",
            "    @%p_out bra DONE;"
        ])

    else:
        # Preserve the mapping for our previous 1D problems
        ptx_lines.extend([
            "    mad.lo.u32 %r_a0, %r_bid, %r_bdim, %r_tid;",
            f"    setp.ge.u32 %p_out, %r_a0, {axis.extent.as_static()};",
            "    @%p_out bra DONE;"
        ])

    if len(tile.axes) == 1 and axis.name != "a0":
        ptx_lines.append(
            f"    mov.u32 %r_{axis.name}, %r_a0;"
        )

    ptx_lines.extend([
        "    and.b32 %r_lane, %r_tid, 31;",
        "    shr.u32 %r_warp, %r_tid, 5;"
    ])

    # Add kernel parameter pointers
    for inst in pointer_instructions:
        ptx_lines.append(f"    {inst}")

    # Add memory loads, arithmetic and output stores
    for inst in body_instructions:
        ptx_lines.append(f"    {inst}")

    # Finish the kernel
    ptx_lines.extend([
        "DONE:",
        "    ret;",
        "}"
    ])

    # Print complete PTX to standard output
    return "\n".join(ptx_lines) + "\n"
