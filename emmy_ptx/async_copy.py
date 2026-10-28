def make_emit_cp_async(instructions, compiler_expr, kernel_meta, arg_order, shared_shapes, body_instructions, use_dynamic_shared, shared_offsets):
    def emit_cp_async(copy):
        if copy.swizzle != "NONE":
            raise NotImplementedError(
                f"Unsupported async swizzle: {copy.swizzle}"
            )

        if copy.nbytes not in (4, 8, 16):
            raise NotImplementedError(
                f"Unsupported async copy size: {copy.nbytes}"
            )

        start = len(instructions)

        src_row = compiler_expr(copy.src_index[0])
        src_col = compiler_expr(copy.src_index[1])

        src_shape = next(
            arg["shape"]
            for arg in kernel_meta["args"]
            if arg["name"] == copy.src
        )

        src_index = f"%r{len(instructions) + 1}"

        instructions.append(
            f"mad.lo.u32 {src_index}, "
            f"{src_row}, {src_shape[1]}, {src_col};"
        )

        dst_row = compiler_expr(copy.smem_index[0])
        dst_col = compiler_expr(copy.smem_index[1])

        if copy.lane_index:
            lane_row = compiler_expr(copy.lane_index[0])
            lane_col = compiler_expr(copy.lane_index[1])

            new_row = f"%r{len(instructions) + 1}"
            instructions.append(
                f"add.u32 {new_row}, {dst_row}, {lane_row};"
            )
            dst_row = new_row

            new_col = f"%r{len(instructions) + 1}"
            instructions.append(
                f"add.u32 {new_col}, {dst_col}, {lane_col};"
            )
            dst_col = new_col

        dst_stride = shared_shapes[copy.smem][1]
        dst_index = f"%r{len(instructions) + 1}"

        instructions.append(
            f"mad.lo.u32 {dst_index}, "
            f"{dst_row}, {dst_stride}, {dst_col};"
        )

        body_instructions.extend(instructions[start:])

        param_index = arg_order.index(copy.src)
        cache_mode = "cg" if copy.nbytes == 16 else "ca"

        smem_base = copy.smem

        if use_dynamic_shared:
            offset = shared_offsets[copy.smem]
            smem_base = (
                "_smem_pool"
                if offset == 0
                else f"_smem_pool+{offset}"
            )
        body_instructions.extend([
            f"mul.wide.u32 %rd_cp_offset, {src_index}, 4;",
            f"add.u64 %rd_cp_addr, "
            f"%rd_base{param_index}, %rd_cp_offset;",
            f"mul.lo.u32 %r_cp_offset, {dst_index}, 4;",
            f"mov.u32 %r_cp_base, {smem_base};",
            "add.u32 %r_cp_addr, %r_cp_base, %r_cp_offset;",
            f"cp.async.{cache_mode}.shared.global "
            f"[%r_cp_addr], [%rd_cp_addr], {copy.nbytes};"
        ])

    return emit_cp_async
